from glob import glob
import os
import os.path as osp
from os.path import join


def get_test_path(test_dir: str):
    print(test_dir)
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