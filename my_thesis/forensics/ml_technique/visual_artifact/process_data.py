import os
import argparse
from collections import defaultdict
import dlib
import cv2
import numpy as np
import pandas as pd
from visual_artifact.pipeline.eyecolor import extract_eyecolor_features
from visual_artifact.pipeline.face_utils import *
from visual_artifact.pipeline import pipeline_utils
from visual_artifact.pipeline.texture import extract_features_eyes,extract_features_faceborder,extract_features_mouth,extract_features_nose
import glob
import random
from PIL import ImageEnhance,Image
import pickle
def load_facedetector():
    """Loads dlib face and landmark detector."""
    dat_file = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/shape_predictor_68_face_landmarks.dat"
    if '/home/phucnp' in os.path.abspath(__file__):
        dat_file = dat_file.replace("/mnt/disk1/doan", "/home")
    # download if missing http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    if not os.path.isfile(dat_file):
        print ('Could not find shape_predictor_68_face_landmarks.dat.')
        exit(-1)
    face_detector = dlib.get_frontal_face_detector()

    sp68 = dlib.shape_predictor(dat_file)
    return face_detector, sp68

face_detector, sp68 = load_facedetector()


final_score_clf = 0.0
final_score_HSV = 0.0
final_feature_vector = None
final_valid_seg = False
scale = 768
def extract_visual_artifact(img):
    face_crop_list, landmarks_list = get_crops_landmarks(face_detector, sp68, img)
    scale = 768
    if (len(face_crop_list) == 1):
        try:
            face_crop = face_crop_list[0]
            landmarks = landmarks_list[0].copy()

            out_size = pipeline_utils.new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
            scale_x = float(out_size[1]) / face_crop.shape[1]
            scale_y = float(out_size[0]) / face_crop.shape[0]

            landmarks_resize = landmarks.copy()
            landmarks_resize[:, 0] = landmarks_resize[:, 0] * scale_x
            landmarks_resize[:, 1] = landmarks_resize[:, 1] * scale_y

            face_crop_resize = cv2.resize(face_crop, (int(out_size[1]), int(out_size[0])), interpolation=cv2.INTER_LINEAR)

            feature_eyecolor, distance_HSV, valid_seg = extract_eyecolor_features(landmarks_resize, face_crop_resize)
            features_eyes = extract_features_eyes(landmarks, face_crop, scale=scale)
            features_mounth = extract_features_mouth(landmarks, face_crop, scale=scale)
            features_nose = extract_features_nose(landmarks, face_crop, scale=scale)
            features_face = extract_features_faceborder(landmarks, face_crop, scale=scale)
            feature = np.concatenate([feature_eyecolor, features_eyes, features_mounth, features_nose, features_face], axis=0)
        except:
            feature = np.array([0])
    else:
        feature = np.array([0])
#     print(feature)
    return feature

from KFold import CustomizeKFold
from tqdm import tqdm
def extract_features(input_real,input_fake, output_path,number_iter):
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

            img = img.astype("uint8")
            contrast = ImageEnhance.Contrast(Image.fromarray(img))
            img = contrast.enhance(1.0)
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1.0)
            img = np.array(img, dtype='uint8')

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

from time import time
from KFold import CustomizeKFold
from glob import glob
from util import *

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
        extract_features(real_path, fake_path, output_fold, -1)
        feature_fold_train.append(output_fold)
    # test
    testset = get_test_path(test_dir=test_dir)
    real_path, fake_path = split_real_fake(train_paths=testset)
    output_fold = feature_ckcpoint + '/test.pkl'
    extract_features(real_path, fake_path, output_fold, -1)
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

