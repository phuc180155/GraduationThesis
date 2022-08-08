#################### VISUALIZE INFERENCE TIME ####################
import numpy as np
import random
from torchsummary import summary
from glob import glob
import torch
import time
import torch.nn as nn
import os
from os.path  import join
from sklearn import metrics
from tqdm import tqdm
from sklearn.svm import SVC
from spectrum.preprocess_data import extract_features as extract_features_spectrum
from visual_artifact.process_data import extract_features as extract_features_visual
import pickle 
from util import get_datasetname, make_outputfeature_dir, split_real_fake, get_test_path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


saved_feature = '/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/test.pkl'

def calculate_metric(y_label, y_pred_label):
    mic_accuracy = accuracy_score(y_label, y_pred_label)
    macro_precision = precision_score(y_label, y_pred_label, average='macro')
    macro_recall = recall_score(y_label, y_pred_label, average='macro')
    macro_f1 = f1_score(y_label, y_pred_label, average='macro')
    return mic_accuracy, macro_precision, macro_recall, macro_f1

def eval_spectrum(data, model_file):
    pkl_file = open(data, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    X = data["data"]
    y = data["label"]
    with open(model_file, 'rb') as f:
        svclassifier_r = pickle.load(f)
    print("Loaded.")
    SVM_score = svclassifier_r.score(X, y)
    print("accuracy by score function: " + str(SVM_score))
    ##### CALCULATE METRIC:
    y_pred = svclassifier_r.predict(X)
    acc, pre, rec, f1 = calculate_metric(y, y_pred)
    # print("accuracy: " + str(acc))
    print("precision: " + str(pre))
    print("recall: " + str(rec))
    print("f1 score: " + str(f1))
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


ckcpoint_spectrum = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/dfdcv5/spectrum/c_2.000000"
saved_feature_spectrum = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/spectrum/output_features/dfdcv5/test.pkl"

for ckcpoint_fold in sorted(glob(join(ckcpoint_spectrum, '*/model.pkl')), key=lambda fold: int(fold.replace('/model.pkl', '')[-1])):
    print("\n********************************************************")
    print("ckcpoint fold: ", ckcpoint_fold)
    eval_spectrum(data=saved_feature_spectrum, model_file=ckcpoint_fold)

ckcpoint_visual = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/ml_technique/checkpoint/model.pkl"
saved_feature_visual_artifact = ""

saved_feature_visual_artifact = ""