import torch
import random
import numpy as np
import os
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from os.path import join
import os.path as osp
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from util import *

def calculate_metric(y_label, y_pred_label):
    mic_accuracy = accuracy_score(y_label, y_pred_label)
    macro_precision = precision_score(y_label, y_pred_label, average='macro')
    macro_recall = recall_score(y_label, y_pred_label, average='macro')
    macro_f1 = f1_score(y_label, y_pred_label, average='macro')
    return mic_accuracy, macro_precision, macro_recall, macro_f1

def eval_visual(data, model_file):
    with open(data, 'rb') as f:
        info_dict = pickle.load(f)
    features_= info_dict['features']
    labels_ = info_dict['labels']

    with open(model_file, 'rb') as f:
        svclassifier_r = pickle.load(f)
    SVM_score = svclassifier_r.score(features_, labels_)
    # print("accuracy by score function: " + str(SVM_score))
    ##### CALCULATE METRIC:
    y_pred = svclassifier_r.predict(features_)
    acc, pre, rec, f1 = calculate_metric(labels_, y_pred)
    # print("accuracy: " + str(acc))
    # print("precision: " + str(pre))
    # print("recall: " + str(rec))
    # print("f1 score: " + str(f1))
    return SVM_score, acc, pre, rec, f1

def extract_test_features(test_dir: str, output_test: str, brightness: float, contrast: float, std=0, missing_value=0, quality=100, type_transform='noise'):
    testset = get_test_path(test_dir=test_dir)
    real_path, fake_path = split_real_fake(train_paths=testset)
    extract_visual_features(real_path, fake_path, output_test, -1, brightness=brightness, contrast=contrast, std=std, miss_size=missing_value, quality=quality, type_transform=type_transform)
    return output_test


test_dir_template = "/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/{}/test"

checkpoints = {
    "deepfake": "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/deepfake/visual_artifact/c_2.000000",
    "faceswap_2d": "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/faceswap_2d/visual_artifact/c_2.000000",
    "3dmm": "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/3dmm/visual_artifact/c_2.000000",
    "faceswap_3d": "//mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/faceswap_3d/visual_artifact/c_2.000000",
    "monkey": "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/monkey/visual_artifact/c_2.000000",
    "reenact": "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/reenact/visual_artifact/c_2.000000",
    "stargan": "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/stargan/visual_artifact/c_2.000000",
    "x2face": "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/x2face/visual_artifact/c_2.000000"
}

brightness = 1.0
contrast = 1.0
stds= [0.0, 0.1, 0.15, 0.2, 0.25]
missing_value = 0.0
quality=100
type_transform = 'noise'
import sys
sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/bash/noise.txt', 'w')
# sys.stdout = sys.__stdout__
for std in stds:
    print("\n*********************************************************")
    print("             std: ", std)
    for technique in ['deepfake', '3dmm', 'faceswap_2d', 'faceswap_3d', 'monkey', 'reenact', 'stargan', 'x2face']:
        if std == 0.0 and technique != 'deepfake':
            continue
        checkpoint = checkpoints[technique]
        print("\n\n ==================================== {} CHECKPOINT =========================".format(technique))

        my_break = False
        # filename = '/contrast{}.pkl'.format(contrast)
        filename = '/noise{}.pkl'.format(std)
        # filename = '/missing{}.pkl'.format(missing_value)
        # filename = '/compression{}.pkl'.format(quality)
        output_test = '/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/visual_artifact/output_features/' + get_datasetname(test_dir_template.format(technique)) + filename
        test_features = extract_test_features(test_dir=test_dir_template.format(technique), output_test=output_test, brightness = brightness, contrast=contrast, std=std, missing_value=missing_value, quality=quality, type_transform=type_transform)        
        
        for pkl_file in sorted(glob(join(checkpoint, '*/*.pkl')), key=lambda pkl: int(pkl.split('/')[-2][-1])):
            if 'model' not in pkl_file:
                continue
                
            fold_id = pkl_file.split('/')[-2][-1]
            ####
            SVM_score, acc, pre, rec, f1 = eval_visual(test_features, pkl_file)
            print("            pt file: ", '/'.join(pkl_file.split('/')[-2:]))
            print("    FOLD: {}    -   ".format(fold_id))
            print("            accuracy =   {:.6f}       |  {:6f}".format(SVM_score, acc))
            print("            precision =  {:.6f}  |   recall =    {:.6f}   |  f1 =    {:.6f}".format(pre, rec, f1))

sys.stdout = sys.__stdout__