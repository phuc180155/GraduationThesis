#################### VISUALIZE INFERENCE TIME ####################
import numpy as np
import random
from torchsummary import summary
import torch
import time
import torch.nn as nn
import os
from sklearn import metrics
from tqdm import tqdm
from sklearn.svm import SVC
from spectrum.preprocess_data import extract_features as extract_features_spectrum
from visual_artifact.process_data import extract_features as extract_features_visual
import pickle 
from util import get_datasetname, make_outputfeature_dir, split_real_fake, get_test_path

saved_feature = '/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/test.pkl'

def extract_features_test_spectrum(test_dir: str):
    print("Extracting...")
    testset = get_test_path(test_dir=test_dir)
    real_path, fake_path = split_real_fake(train_paths=testset)
    extract_features_spectrum(real_path, fake_path, -1, saved_feature)

def eval_spectrum(data, model_file):
    pkl_file = open(data, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    X = data["data"]
    y = data["label"]
    with open(model_file, 'rb') as f:
        svclassifier_r = pickle.load(f)
    SVM_score = svclassifier_r.score(X, y)
    print("accuracy: " + str(SVM_score))
    return SVM_score

def eval_visual(data, model_file):
    with open(data, 'rb') as f:
        info_dict = pickle.load(f)
    features_= info_dict['features']
    labels_ = info_dict['labels']

    with open(model_file, 'rb') as f:
        svclassifier_r = pickle.load(f)
    SVM_score = svclassifier_r.score(features_, labels_)
    print("SVM: " + str(SVM_score))
    return SVM_score

def get_detection_speed_spectrum(ckcpoint_model: str):
    time.sleep(5)
    test_dir = "/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/test"
    begin = time.time()
    extract_features_test_spectrum(test_dir=test_dir)
    # test
    acc = eval_spectrum(data=saved_feature, model_file=ckcpoint_model)
    processing_time = time.time() - begin
    print('acc: ', acc)
    print('processing time: ', processing_time)
    return processing_time

def extract_features_test_visual_artifact(test_dir: str):
    print("Extracting...")
    # test
    testset = get_test_path(test_dir=test_dir)
    real_path, fake_path = split_real_fake(train_paths=testset)
    extract_features_visual(real_path, fake_path, saved_feature, -1)

def get_detection_speed_visual_artifact(ckcpoint_model: str):
    time.sleep(5)
    test_dir = "/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/test"
    begin = time.time()
    extract_features_test_visual_artifact(test_dir=test_dir)
    # test
    acc = eval_visual(data=saved_feature, model_file=ckcpoint_model)
    processing_time = time.time() - begin
    print('acc: ', acc)
    print('processing time: ', processing_time)
    return processing_time

ckcpoint_spectrum = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/deepfake/spectrum/c_2.000000/0.9876_fold_0/model.pkl"
ckcpoint_visual = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/model.pkl"
get_detection_speed_visual_artifact(ckcpoint_model=ckcpoint_visual)