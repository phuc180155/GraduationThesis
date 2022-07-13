from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import numpy as np
from headpose_forensic.process_data import extract_features_kfold
from headpose_forensic.eval_headpose import eval_headposes
import os.path as osp
from os.path import join
import os

def process_training_data(data):
    videos_real = []
    videos_fake = []
    video_list = []
    label_list = []

    R_vec_feat = []
    R_mat_feat = []
    R_mat_full_feat = []
    t_vec_feat = []

    for key, value in data.items():
        label = value['label']
        if label == 'real':
            label_id = 0
            videos_real.append(key)
        else:
            label_id = 1
            videos_fake.append(key)

        # print(key)
        R_c_list = value['R_c_vec']
        R_c_matrix_list = value['R_c_mat']
        t_c_list = value['t_c']

        R_a_list = value['R_a_vec']
        R_a_matrix_list = value['R_a_mat']
        t_a_list = value['t_a']

        # Compute diff
        delta_R_vec_list = [R_c_list[i][:, -1] - R_a_list[i][:, -1] for i in range(len(R_c_list)) if R_c_list[i] is not None]
        delta_t_vec_list = [t_c_list[i][:, -1] - t_a_list[i][:, -1] for i in range(len(t_c_list)) if t_c_list[i] is not None]
        delta_R_mat_list = [R_c_matrix_list[i][:, -1] - R_a_matrix_list[i][:, -1] for i in range(len(R_c_matrix_list)) if R_c_matrix_list[i] is not None]
        delta_R_full_mat_list = [(R_c_matrix_list[i] - R_a_matrix_list[i]).flatten() for i in range(len(R_c_matrix_list)) if R_c_matrix_list[i] is not None]

        R_vec_feat += delta_R_vec_list
        R_mat_feat += delta_R_mat_list
        t_vec_feat += delta_t_vec_list
        R_mat_full_feat += delta_R_full_mat_list

        label_list += [label_id] * len(delta_R_mat_list)
        video_list += [key] * len(delta_R_mat_list)

    return sorted(set(videos_real)), sorted(set(videos_fake)), video_list, label_list, R_vec_feat, R_mat_feat, R_mat_full_feat, t_vec_feat

def train_model(features, label_list, random_state=0, C=1):
    X_train, y_train = shuffle(features, label_list, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = svm.SVC(C=C, kernel='rbf', gamma=0.86)
    clf.fit(X_train, y_train)
    return clf, scaler

from util import inspect_preprocess, make_ckcpoint

def train_headpose(model_name: str, train_dir: str, val_dir: str, test_dir: str, checkpoint: str, n_folds: int, use_trick: int, C: float):
    feature_fold_train, feature_fold_test = extract_features_kfold(model_name=model_name, n_folds=n_folds, use_trick=use_trick, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)
    #
    for fold_idx in range(n_folds):         ## train
        pkl_file = open(feature_fold_train[fold_idx], 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        videos_real, videos_fake, video_list, label_list, R_vec_feat, R_mat_feat, R_mat_full_feat, t_vec_feat = process_training_data(data)
        features = [np.concatenate([R_mat_full_feat[i], t_vec_feat[i]]) for i in range(len(R_mat_feat))]
        classifier, scaler = train_model(features, label_list, C=C)
        model = [classifier, scaler]
        #
        fold_ckcpoint = make_ckcpoint(checkpoint=checkpoint, fold_idx=fold_idx, C=C)
        output_model_fold = join(fold_ckcpoint, 'model.pkl')
        with open(output_model_fold, 'wb') as f:
            pickle.dump(model, f)
        # test
        acc_fold = eval_headposes(data=feature_fold_test, model_file=output_model_fold)
        os.rename(src=fold_ckcpoint, dst=join("/".join(fold_ckcpoint.split("/")[:-1]), '{}_fold_{}'.format(acc_fold, fold_idx)))


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data', dest='input',default='',
                        help='Path to pkl data file.')
    parser.add_argument('-o', '--model', dest='output', help='path to save model.')
    args = parser.parse_args()
    return args

if __name__ == "__name__":
    args_in = parse_args()
    train_headpose(args_in.data,args_in.model)