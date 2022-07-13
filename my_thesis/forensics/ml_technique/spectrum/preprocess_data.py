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

data= {}
epsilon = 1e-8
N = 300
y = []
error = []
random.seed(0)


def extract_features(input_real,input_fake,number_iter,output_path):
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

        img = img.astype("uint8")
        contrast = ImageEnhance.Contrast(Image.fromarray(img))
        img = contrast.enhance(1.0)
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.0)
        img = np.array(img, dtype='float64')

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
    print("Error real files: ", real_error)

    if number_iter == -1:
        number_iter = len(input_fake)
    psd1D_total2 = np.zeros([number_iter, N])
    label_total2 = np.zeros([number_iter])

    cont = 0

    for file in tqdm(rootdirs[1]):
        filename = file
        parts = filename.split("/")

        img = cv2.imread(filename, 0)

        img = img.astype("uint8")
        contrast = ImageEnhance.Contrast(Image.fromarray(img))
        img = contrast.enhance(1.0)
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.0)
        img = np.array(img, dtype='float64')

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
    print("Error fake files: ", fake_error)

    psd1D_total_final = np.concatenate((psd1D_total,psd1D_total2), axis=0)
    label_total_final = np.concatenate((label_total,label_total2), axis=0)

    data["data"] = psd1D_total_final
    data["label"] = label_total_final

    output = open(output_path, 'wb')
    pickle.dump(data, output)
    output.close()
    print("DATA Saved")

from time import time
from util import get_datasetname, make_outputfeature_dir, split_real_fake, get_test_path

def extract_features_kfold(model_name: str, n_folds: int, use_trick: int, train_dir: str, val_dir: str, test_dir: str):
    kfold = CustomizeKFold(n_folds=n_folds, train_dir=train_dir, val_dir=val_dir, trick=use_trick)
    datasetname = get_datasetname(train_dir)
    feature_ckcpoint = make_outputfeature_dir(model_name, datasetname)
    # train
    print("Extracting...")
    begin = time()
    feature_fold_train, feature_fold_test = [], []
    files = os.listdir(feature_ckcpoint)
    if len(files):
        feature_fold_train = [feature_ckcpoint + '/train_fold{}.pkl'.format(fold_idx) for fold_idx in range(n_folds)]
        feature_fold_test = feature_ckcpoint + '/test.pkl'
        return feature_fold_train, feature_fold_test

    for fold_idx in range(n_folds):
        print("          *****: FOLD {} ".format(fold_idx))
        trainset, valset = kfold.get_fold(fold_idx=fold_idx)
        real_path, fake_path = split_real_fake(train_paths=trainset)
        output_fold = feature_ckcpoint + '/train_fold{}.pkl'.format(fold_idx)
        extract_features(real_path, fake_path, -1, output_fold)
        feature_fold_train.append(output_fold)
    # test
    testset = get_test_path(test_dir=test_dir)
    real_path, fake_path = split_real_fake(train_paths=testset)
    output_fold = feature_ckcpoint + '/test.pkl'
    extract_features(real_path, fake_path, -1, output_fold)
    print("Extracted: ", time() - begin)
    feature_fold_test = output_fold
    return feature_fold_train, feature_fold_test


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-ir', '--input_real', dest='input_real',default='',
                        help='Path to input image or folder containting multiple images.')
    parser.add_argument('-if', '--input_fake', dest='input_fake',default='',
                        help='Path to input image or folder containting multiple images.')
    parser.add_argument('-o', '--output', dest='output', help='Path to save outputs.',
                        default='./output')
    parser.add_argument('-n', '--number_iter', default=100,help='number image process')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args_in = parse_args()
    extract_features_kfold(args_in.input_real,args_in.input_fake, int(args_in.number_iter),args_in.output)
