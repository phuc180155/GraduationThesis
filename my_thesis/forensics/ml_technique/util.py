from glob import glob
import os
import os.path as osp
from os.path import join
import cv2
import numpy as np
import os, pickle, argparse
from spectrum import radialProfile
from glob import glob
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import griddata
from PIL import ImageEnhance, Image
import random
from tqdm import tqdm
from KFold import CustomizeKFold
from albumentations.augmentations.transforms import ImageCompression
from albumentations import Compose,Normalize
from visual_artifact.process_data import extract_visual_artifact


def get_test_path(test_dir: str):
    # print(test_dir)
    return list(glob(os.path.join(test_dir, '*/*')))

def split_real_fake(train_paths: str):
    real, fake = [], []
    for path in train_paths:
        cls = path.split('/')[-2]
        if cls not in ['0_real', '1_fake', '1_df']:
            print("bug appear! ", path)
            exit(0)
        if 'real' in cls:
            real.append(path)
        else:
            fake.append(path)
    return real, fake

def get_datasetname(dir: str):
    if 'dfdcv5' in dir:
        return 'dfdcv5'
    if 'dfdcv6' in dir:
        return 'dfdcv6'
    if 'Celeb-DFv6' in dir:
        return 'Celeb-DFv6'
    if 'df_in_the_wildv6' in dir:
        return 'df_in_the_wildv6'
    if 'UADFV' in dir:
        return 'UADFV'
    if '3dmm' in dir:
        return '3dmm'
    if 'deepfake' in dir:
        return 'deepfake'
    if 'faceswap_2d' in dir:
        return 'faceswap_2d'
    if 'faceswap_3d' in dir:
        return 'faceswap_3d'
    if 'monkey' in dir:
        return 'monkey'
    if 'reenact' in dir:
        return 'reenact'
    if 'stargan' in dir:
        return 'stargan'
    if 'x2face' in dir:
        return 'x2face'
    if 'ff' in dir:
        if 'component' in dir:
            return 'ff_' + dir.split('/')[-2]
        if 'all' in dir:
            return 'ff_all'

def make_outputfeature_dir(model: str, datasetname: str):
    dir1 = model + "/output_features"
    dir2 = model + "/output_features/" + datasetname
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    return dir2

def inspect_preprocess(dir: str):
    datasetname = get_datasetname(dir=dir)
    files = os.listdir("output_features/" + datasetname)
    return bool(len(files))

def make_ckcpoint(checkpoint: str, fold_idx: int, C: float):
    if not osp.exists(checkpoint):
        os.mkdir(checkpoint)

    ckcpoint1 = checkpoint + "/c_{:4f}".format(C)
    ckcpoint2 = checkpoint + "/c_{:4f}/".format(C) + 'fold_{}'.format(fold_idx)
    if not osp.exists(ckcpoint1):
        os.mkdir(ckcpoint1)
    if not osp.exists(ckcpoint2):
        os.mkdir(ckcpoint2)
    return ckcpoint2


from torchvision import datasets, transforms
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def my_transform():
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=1.0),
        # transforms.RandomRotation(20, expand=True),
        # transforms.RandomAffine(degrees=20, scale=(0.5, 1.5)),
        # transforms.ColorJitter(brightness=1.0, contrast=0.0, saturation=0.0, hue=0.0),
        # transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=1.0, hue=0.0),
        # transforms.RandomApply([
        #     transforms.ToTensor(),
        #     AddGaussianNoise(mean=0.2, std=0.1),
        #     transforms.ToPILImage()
        # ],p=1.0),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, ratio=(0.99999, 0.999999), scale=(0.05, 0.05+1e-5), value=1.),
            transforms.ToPILImage()
        ])
    ])

def get_jpeg_augmentation(quality):
    train_transform = [
        ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)
    ]
    transforms =  Compose(train_transform)
    return lambda img:Image.fromarray(transforms(image=np.array(img))['image'])

def transform_compression(path, quality):
    transform = transforms.Compose([
        get_jpeg_augmentation(quality)
    ])
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("uint8")
    img = Image.fromarray(img)
    transform = get_jpeg_augmentation(quality)
    img = transform(img)
    return np.array(img, dtype='uint8')

def transform_noise(path, std):
    transform = transforms.Compose([
        transforms.RandomApply([
            transforms.ToTensor(),
            AddGaussianNoise(mean=0, std=std),
            transforms.ToPILImage()
        ],p=1.0),
    ])
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("uint8")
    img = Image.fromarray(img)
    img = transform(img)
    return np.array(img, dtype='uint8')

def transform_missing_value(path, size):
    transform = transforms.Compose([
        transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, ratio=(0.999999999, 0.99999999999), scale=(size, size+1e-8), value=0.),
            transforms.ToPILImage()
        ])
    ])
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("uint8")
    img = Image.fromarray(img)
    img = transform(img)
    return np.array(img, dtype='uint8')

def transform_contrast_brightness(path, brightness, contrast):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("uint8")
    img = Image.fromarray(img)
    contrast_transform = ImageEnhance.Contrast(img)
    img = contrast_transform.enhance(contrast)
    brightness_transform = ImageEnhance.Brightness(img)
    img = brightness_transform.enhance(brightness)
    img = np.array(img, dtype='uint8')
    return img

def extract_visual_features(input_real, input_fake, output_path, number_iter, brightness=1, contrast=1, std=0, miss_size=0, quality=100, type_transform='contrast_brightness'):
    features = []
    labels = []
    cont = 0
    video_dir_dict = {}
    video_dir_dict['real'] = input_real
    video_dir_dict['fake'] = input_fake
    limited_samples = True if number_iter != -1 else False
    number_iter_real = len(input_real) if number_iter == -1 else number_iter
    number_iter_fake = len(input_fake) if number_iter == -1 else number_iter

    for tag in video_dir_dict:
        cont = 0
        if tag == 'real':
            label = 0
        else:
            label = 1
        input_path = video_dir_dict[tag]
        number_iter = number_iter_real if tag == 'real' else number_iter_fake

        random.shuffle(input_path)
        cannot_extract = 0
        for vid_path in tqdm(input_path):
            img = cv2.imread(vid_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Transform:
            if type_transform == 'contrast_brightness':
                img = transform_contrast_brightness(vid_path, brightness, contrast)
            elif type_transform == 'noise':
                img = transform_noise(vid_path, std)
            elif type_transform == 'missing_value':
                img = transform_missing_value(vid_path, miss_size)
            elif type_transform == 'compression':
                img = transform_compression(vid_path, quality)
            
            # img = img.astype("uint8")
            # contrast = ImageEnhance.Contrast(Image.fromarray(img))
            # img = contrast.enhance(1.0)
            # brightness = ImageEnhance.Brightness(img)
            # img = brightness.enhance(1.0)
            # img = np.array(img, dtype='uint8')

            feature = extract_visual_artifact(img)
            if len(feature) < 3:
                cannot_extract += 1
                continue
            features.append(feature)
            labels.append([label])
            cont += 1
        print("Cannot extract images: ", cannot_extract)

    info_dict = {'features':features,'labels':labels}
    # print(info_dict)
    with open(output_path, 'wb') as f:
        pickle.dump(info_dict, f)
    
def extract_spectrum_features(input_real, input_fake, number_iter, output_path, brightness=1, contrast=1, std=0, miss_size=0, quality=100, type_transform='contrast_brightness'):
    data= {}
    epsilon = 1e-8
    N = 300
    y = []
    error = []
    random.seed(0)

    if number_iter == -1:
        number_iter = len(input_real)
        get_all = True

    psd1D_total = np.zeros([number_iter, N])
    label_total = np.zeros([number_iter])
    cont = 0

    # real data
    rootdirs = [input_real,input_fake]
    real_error, fake_error = 0, 0
    for file in tqdm(rootdirs[0]):
        filename = file
        img = cv2.imread(filename, 0)

        # Transform:
        if type_transform == 'contrast_brightness':
            img = transform_contrast_brightness(filename, brightness, contrast)
        elif type_transform == 'noise':
            img = transform_noise(filename, std)
        elif type_transform == 'missing_value':
            img = transform_missing_value(filename, miss_size)
        elif type_transform == 'compression':
            img = transform_compression(filename, quality)

        # img = img.astype("uint8")
        # contrast = ImageEnhance.Contrast(Image.fromarray(img))
        # img = contrast.enhance(1.0)
        # brightness = ImageEnhance.Brightness(img)
        # img = brightness.enhance(1.0)
        # img = np.array(img, dtype='float64')
        #######

        # we crop the center
        h = int(img.shape[0] / 3)
        w = int(img.shape[1] / 3)
        img = img[h:-h, w:-w]

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fshift += epsilon
        try:
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            # Calculate the azimuthally averaged 1D power spectrum
            points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
            xi = np.linspace(0, N, num=N)  # coordinates for interpolation

            interpolated = griddata(points, psd1D, xi, method='cubic')
            interpolated /= interpolated[0]

            psd1D_total[cont, :] = interpolated
            label_total[cont] = 0
            cont += 1
        #             print(cont)
        except:
            real_error += 1
        if cont == number_iter and not get_all:
            break
    # print("Error real files: ", real_error)

    if number_iter == -1:
        number_iter = len(input_fake)
    psd1D_total2 = np.zeros([number_iter, N])
    label_total2 = np.zeros([number_iter])

    cont = 0

    for file in tqdm(rootdirs[1]):
        filename = file
        parts = filename.split("/")

        # Transform:
        if type_transform == 'contrast_brightness':
            img = transform_contrast_brightness(filename, brightness, contrast)
        elif type_transform == 'noise':
            img = transform_noise(filename, std)
        elif type_transform == 'missing_value':
            img = transform_missing_value(filename, miss_size)
        elif type_transform == 'compression':
            img = transform_compression(filename, quality)
        # img = cv2.imread(filename, 0)
        # img = img.astype("uint8")
        # contrast = ImageEnhance.Contrast(Image.fromarray(img))
        # img = contrast.enhance(1.0)
        # brightness = ImageEnhance.Brightness(img)
        # img = brightness.enhance(1.0)
        # img = np.array(img, dtype='float64')

        # we crop the center
        h = int(img.shape[0] / 3)
        w = int(img.shape[1] / 3)
        img = img[h:-h, w:-w]

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fshift += epsilon

        try:
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            # Calculate the azimuthally averaged 1D power spectrum
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
            xi = np.linspace(0, N, num=N)  # coordinates for interpolation

            interpolated = griddata(points, psd1D, xi, method='cubic')
            interpolated /= interpolated[0]

            psd1D_total2[cont, :] = interpolated
            label_total2[cont] = 1
            cont += 1
        #             print(cont)
        except:
            fake_error += 1
        if cont == number_iter and not get_all:
            break
    # print("Error fake files: ", fake_error)

    psd1D_total_final = np.concatenate((psd1D_total,psd1D_total2), axis=0)
    label_total_final = np.concatenate((label_total,label_total2), axis=0)

    data["data"] = psd1D_total_final
    data["label"] = label_total_final

    output = open(output_path, 'wb')
    pickle.dump(data, output)
    output.close()
    # print("DATA Saved")