import numpy as np
import cv2
import random
from typing import List, Dict
from glob import glob
from tqdm import tqdm
from os.path import join
import os.path as osp

np.random.seed(0)
random.seed(0)

class CustomizeKFold(object):
    def __init__(self, n_folds: int, train_dir: str, val_dir: str, trick=True):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.n_folds = n_folds
        self.trick = trick
        if self.trick:
            if 'dfdcv5' in train_dir:
                self.do_trick = 'real:5000,fake:2000'
            if 'dfdcv6' in train_dir:
                self.do_trick = 'real:0,fake:0'
            if 'df_in_the_wild' in train_dir:
                if 'df_in_the_wildv5' in train_dir:
                    self.do_trick = 'real:4000,fake:2000'
                if 'df_in_the_wildv6' in train_dir:
                    self.do_trick = 'real:0,fake:0'
            if 'Celeb-DFv5' in train_dir:
                self.do_trick = 'real:5000,fake:2000'
            if 'Celeb-DFv6' in train_dir:
                self.do_trick = 'real:0,fake:0'
            if 'UADFV' in train_dir:
                self.do_trick = 'real:0,fake:0'
            if 'ff' in train_dir:
                self.do_trick = 'real:0,fake:0'
            if 'extend_data_train' in train_dir:
                self.do_trick = 'real:0,fake:0'

        # Concat train and val
        self.real_paths, self.fake_paths = self.get_path()
        print("Number of real images: ", len(self.real_paths))
        print("Number of fake images: ", len(self.fake_paths))
        self.real_videos, self.fake_videos = self.extract_video()
        self.real_video_names = list(self.real_videos.keys())
        self.fake_video_names = list(self.fake_videos.keys())
        self.num_real_videos, self.num_fake_videos = len(self.real_video_names), len(self.fake_video_names)
        random.shuffle(self.real_video_names)
        random.shuffle(self.fake_video_names)
        print("real video: ", self.num_real_videos)
        # print("Sample list real: ", self.real_video_names[0], self.real_video_names[-1])
        # print("Sample list dict real: ", (self.real_video_names[0], self.real_videos[self.real_video_names[0]]), (self.real_video_names[-1], self.real_videos[self.real_video_names[-1]]))
        print("fake video: ", self.num_fake_videos)
        # print("Sample list fake: ", self.fake_video_names[0], self.fake_video_names[-1])
        # print("Sample list dict fake: ", (self.fake_video_names[0], self.fake_videos[self.fake_video_names[0]]), (self.fake_video_names[-1], self.fake_videos[self.fake_video_names[-1]]))


    def get_path(self):
        cls_real = ['0_real']
        cls_fake = ['1_df', '1_f2f', '1_fs', '1_nt', '1_fake']
        real_paths, fake_paths = [], []
        for c in cls_real:
            real_paths.extend(glob(join(self.train_dir, '{}/*'.format(c))))
            real_paths.extend(glob(join(self.val_dir, '{}/*'.format(c))))
        for c in cls_fake:
            fake_paths.extend(glob(join(self.train_dir, '{}/*'.format(c))))
            fake_paths.extend(glob(join(self.val_dir, '{}/*'.format(c))))
        return real_paths, fake_paths

    def extract_video(self):
        """ Return real_videos: {video1: [video1_0, video1_1... video1_n], video2: [video2_0, ... video2_m], ...}
        """
        def extract(paths: List[str]):
            result = {}
            for p in paths:
                info = osp.basename(p).split('_')
                video_name = '_'.join(info[:-1])
                image_idx = info[-1]
                if video_name not in result.keys():
                    result[video_name] = [p]
                else:
                    result[video_name].append(p)
            return result
        real_videos = extract(paths=self.real_paths)
        fake_videos = extract(paths=self.fake_paths)
        return real_videos, fake_videos

    def get_fold(self, fold_idx: int):
        """ Trả về tập train_path, val_path
        """
        #
        if 'extend_data_train' in self.train_dir:
            return self.get_fold_in_extenddata(fold_idx=fold_idx)
        # 
        readfile, train_file, val_file, prefix_old, prefix_new = self.inspect_must_read_files(fold_idx=fold_idx)
        if readfile:
            print("Reading images from files...")
            train_images = self.read_images_from_file(file=train_file, prefix_old=prefix_old, prefix_new=prefix_new)
            val_images = self.read_images_from_file(file=val_file, prefix_old=prefix_old, prefix_new=prefix_new)
            return train_images, val_images

        # Get fold in real class
        train_real_videos, val_real_videos = self.get_fold_in_cls(fold_idx=fold_idx, cls='real')
        train_real_images, val_real_images = [], []
        for v in train_real_videos:
            train_real_images.extend(self.real_videos[v])
        for v in val_real_videos:
            val_real_images.extend(self.real_videos[v])

        # print("get fold: ")
        # print("train real: ", len(train_real_images))
        # print("val real: ", len(val_real_images))

        # Get fold in fake class
        train_fake_videos, val_fake_videos = self.get_fold_in_cls(fold_idx=fold_idx, cls='fake')
        train_fake_images, val_fake_images = [], []
        for v in train_fake_videos:
            train_fake_images.extend(self.fake_videos[v])
        for v in val_fake_videos:
            val_fake_images.extend(self.fake_videos[v])

        # print("train fake: ", len(train_fake_images))
        # print("val fake: ", len(val_fake_images))
        # trick:
        if self.trick:
            random.shuffle(train_real_images)
            random.shuffle(train_fake_images)
            info = self.do_trick.split(',')
            num_real = int(info[0].split(':')[-1])
            val_real_images.extend(train_real_images[:num_real])
            train_real_images = train_real_images[num_real:]
            num_fake = int(info[1].split(':')[-1])
            val_fake_images.extend(train_fake_images[:num_fake])
            train_fake_images = train_fake_images[num_fake:]
            # print("After do trick")
            # print("train real: ", len(train_real_images))
            # print("val real: ", len(val_real_images))
            # print("train fake: ", len(train_fake_images))
            # print("val fake: ", len(val_fake_images))
            # print("test 10 train images: ")
        # Concatenate:
        train_images = train_real_images + train_fake_images
        val_images = val_real_images + val_fake_images
        # for i in range(10):
        #     print("     ", train_images[i])
        # print("test 10 val images:")
        # for i in range(10):
        #     print("     ", val_images[i])
        return train_images, val_images

    def get_fold_in_cls(self, fold_idx: int, cls='real'):
        if cls == 'real':
            num_samples = self.num_real_videos
            samples = np.array(self.real_video_names, dtype=object)
        else:
            num_samples = self.num_fake_videos
            samples = np.array(self.fake_video_names, dtype=object)
        num_samples_per_fold = num_samples // self.n_folds
        val_idx_from = num_samples_per_fold * fold_idx
        val_idx_to = num_samples_per_fold * (fold_idx + 1) if (fold_idx != self.n_folds - 1) else num_samples
        val = samples[val_idx_from: val_idx_to]
        train = np.concatenate([samples[:val_idx_from], samples[val_idx_to:]])
        return train, val

    def get_fold_in_extenddata(self, fold_idx: int):
        """ Trả về tập train_path, val_path
        """
        # 
        readfile, train_file, val_file, prefix_old, prefix_new = self.inspect_must_read_files(fold_idx=fold_idx)
        if readfile:
            print("Reading images from files...")
            train_images = self.read_images_from_file(file=train_file, prefix_old=prefix_old, prefix_new=prefix_new)
            val_images = self.read_images_from_file(file=val_file, prefix_old=prefix_old, prefix_new=prefix_new)
            return train_images, val_images

        # Get fold in real class
        train_real_images, val_real_images = self.get_fold_by_cls_in_extendata(fold_idx=fold_idx, cls='real')

        print("get fold: ")
        print("train real: ", len(train_real_images))
        print("val real: ", len(val_real_images))

        # Get fold in fake class
        train_fake_images, val_fake_images = self.get_fold_by_cls_in_extendata(fold_idx=fold_idx, cls='fake')

        print("train fake: ", len(train_fake_images))
        print("val fake: ", len(val_fake_images))
        # trick:
        if self.trick:
            random.shuffle(train_real_images)
            random.shuffle(train_fake_images)
            info = self.do_trick.split(',')
            num_real = int(info[0].split(':')[-1])
            val_real_images.extend(train_real_images[:num_real])
            train_real_images = train_real_images[num_real:]
            num_fake = int(info[1].split(':')[-1])
            val_fake_images.extend(train_fake_images[:num_fake])
            train_fake_images = train_fake_images[num_fake:]
            # print("After do trick")
            # print("train real: ", len(train_real_images))
            # print("val real: ", len(val_real_images))
            # print("train fake: ", len(train_fake_images))
            # print("val fake: ", len(val_fake_images))
            # print("test 10 train images: ")
        # Concatenate:
        train_images = train_real_images + train_fake_images
        val_images = val_real_images + val_fake_images
        # for i in range(10):
        #     print("     ", train_images[i])
        # print("test 10 val images:")
        # for i in range(10):
        #     print("     ", val_images[i])
        return train_images, val_images

    def get_fold_by_cls_in_extendata(self, fold_idx: int, cls='real'):
        if cls == 'real':
            num_samples = len(self.real_paths)
            samples = np.array(self.real_paths, dtype=object)
        else:
            num_samples = len(self.fake_paths)
            samples = np.array(self.fake_paths, dtype=object)
        num_samples_per_fold = num_samples // self.n_folds
        val_idx_from = num_samples_per_fold * fold_idx
        val_idx_to = num_samples_per_fold * (fold_idx + 1) if (fold_idx != self.n_folds - 1) else num_samples
        val = samples[val_idx_from: val_idx_to]
        train = np.concatenate([samples[:val_idx_from], samples[val_idx_to:]])
        return train.tolist(), val.tolist()

    def read_images_from_file(self, file: str, prefix_old: str, prefix_new: str):
        imgs = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img = line.strip().replace(prefix_old, prefix_new)
                if not osp.exists(img):
                    print('bug: ', img)
                else:
                    imgs.append(img)
        return imgs

    def inspect_must_read_files(self, fold_idx: str):
        datasetname = self.get_datasetname()
        curdev = self.get_curdevice()
        dataset_pos = {
            'dfdcv5': '61',
            'dfdcv6': '8',
            'celeb_dfv5': '8',
            'celeb_dfv6': '61',
            'wildv5': '8',
            'wildv6': '8',
            'uadfv': '61',
            '3dmm': '61',
            'deepfake': '61',
            'faceswap_2d': '61',
            'faceswap_3d': '61',
            'monkey': '61',
            'reenact': '61',
            'stargan': '61',
            'x2face': '61',
            'ff_df': '61',
            'ff_f2f': '61',
            'ff_fs': '61',
            'ff_nt': '61',
            'ff_all': '61'
        }
        if dataset_pos[datasetname] == curdev:
            return False, 'trainfile', 'valfile', '', ''
        else:
            if datasetname == 'dfdcv5':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61dfdcv5/train/fold_{}.txt'.format(fold_idx), 'inspect/61dfdcv5/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'dfdcv6':
                prefix_new = '/mnt/disk1/doan/'
                prefix_old = '/home/'
                return True, 'inspect/8dfdcv6/train/fold_{}.txt'.format(fold_idx), 'inspect/8dfdcv6/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'celeb_dfv6':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61celebdfv6/train/fold_{}.txt'.format(fold_idx), 'inspect/61celebdfv6/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'wildv5':
                prefix_new = '/mnt/disk1/doan/'
                prefix_old = '/home/'
                return True, 'inspect/8wildv5/train/fold_{}.txt'.format(fold_idx), 'inspect/8wildv5/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'celeb_dfv5':
                prefix_new = '/mnt/disk1/doan/'
                prefix_old = '/home/'
                return True, 'inspect/8celebdfv5/train/fold_{}.txt'.format(fold_idx), 'inspect/8celebdfv5/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'wildv6':
                prefix_new = '/mnt/disk1/doan/'
                prefix_old = '/home/'
                return True, 'inspect/8wildv6/train/fold_{}.txt'.format(fold_idx), 'inspect/8wildv6/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'uadfv':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61uadfv/train/fold_{}.txt'.format(fold_idx), 'inspect/61uadfv/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == '3dmm':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/613dmm/train/fold_{}.txt'.format(fold_idx), 'inspect/613dmm/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'deepfake':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61deepfake/train/fold_{}.txt'.format(fold_idx), 'inspect/61deepfake/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'faceswap_2d':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61faceswap_2d/train/fold_{}.txt'.format(fold_idx), 'inspect/61faceswap_2d/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'faceswap_3d':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61faceswap_3d/train/fold_{}.txt'.format(fold_idx), 'inspect/61faceswap_3d/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'monkey':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61monkey/train/fold_{}.txt'.format(fold_idx), 'inspect/61monkey/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'reenact':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61reenact/train/fold_{}.txt'.format(fold_idx), 'inspect/61reenact/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'stargan':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61stargan/train/fold_{}.txt'.format(fold_idx), 'inspect/61stargan/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'x2face':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61x2face/train/fold_{}.txt'.format(fold_idx), 'inspect/61x2face/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'ff_df':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61ff_df/train/fold_{}.txt'.format(fold_idx), 'inspect/61ff_df/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'ff_f2f':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61ff_f2f/train/fold_{}.txt'.format(fold_idx), 'inspect/61ff_f2f/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'ff_fs':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61ff_fs/train/fold_{}.txt'.format(fold_idx), 'inspect/61ff_fs/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'ff_nt':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61ff_nt/train/fold_{}.txt'.format(fold_idx), 'inspect/61ff_nt/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new
            if datasetname == 'ff_all':
                prefix_old = '/mnt/disk1/doan/'
                prefix_new = '/home/'
                return True, 'inspect/61ff_all/train/fold_{}.txt'.format(fold_idx), 'inspect/61ff_all/val/fold_{}.txt'.format(fold_idx), prefix_old, prefix_new

    def get_curdevice(self):
        return '61' if '/mnt/disk1/doan' in self.train_dir else '8'

    def get_datasetname(self):
        if 'dfdcv5' in self.train_dir:
            return 'dfdcv5'
        if 'dfdcv6' in self.train_dir:
            return 'dfdcv6'
        if 'Celeb-DFv5' in self.train_dir:
            return 'celeb_dfv5'
        if 'df_in_the_wildv5' in self.train_dir:
            return 'wildv5'
        if 'df_in_the_wildv6' in self.train_dir:
            return 'wildv6'
        if 'Celeb-DFv6' in self.train_dir:
            return 'celeb_dfv6'
        if 'UADFV' in self.train_dir:
            return 'uadfv'
        if '3dmm' in self.train_dir:
            return '3dmm'
        if 'deepfake' in self.train_dir:
            return 'deepfake'
        if 'faceswap_2d' in self.train_dir:
            return 'faceswap_2d'
        if 'faceswap_3d' in self.train_dir:
            return 'faceswap_3d'
        if 'monkey' in self.train_dir:
            return 'monkey'
        if 'reenact' in self.train_dir:
            return 'reenact'
        if 'stargan' in self.train_dir:
            return 'stargan'
        if 'x2face' in self.train_dir:
            return 'x2face'
        if 'ff' in self.train_dir:
            if 'all' in self.train_dir:
                return 'ff_all'
            if 'component' in self.train_dir:
                if 'Deepfakes' in self.train_dir:
                    return 'ff_df'
                if 'Face2Face' in self.train_dir:
                    return 'ff_f2f'
                if 'FaceSwap' in self.train_dir:
                    return 'ff_fs'
                if 'NeuralTexture' in self.train_dir:
                    return 'ff_nt'
        return ''
        

if __name__ == '__main__':
    customize_kfold = CustomizeKFold(n_folds=5, train_dir='/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv5/image/train', val_dir='/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv5/image/val', trick=True)
    # for i in range(5):
    #     train, val = customize_kfold.get_fold(i)
    #     print("Fold: ", i)
    #     print("     train: ", len(train))
    #     print("     val: ", len(val))