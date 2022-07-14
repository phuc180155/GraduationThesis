import os
import argparse
import pandas as pd
from sklearn.svm import SVC

import pickle
import os
import os.path as osp
import sys
from os.path import join
from visual_artifact.process_data import extract_features_kfold
from visual_artifact.eval_visual import eval_visual
from util import make_ckcpoint

def train_visual(model_name: str, train_dir: str, val_dir: str, test_dir: str, checkpoint: str, n_folds: int, use_trick: int, C: float):
    feature_fold_train, feature_fold_test = extract_features_kfold(model_name=model_name, n_folds=n_folds, use_trick=use_trick, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)
    #
    for fold_idx in range(n_folds):
        ## train
        svclassifier_r = SVC(C=C)
        with open(feature_fold_train[fold_idx], 'rb') as f:
            info_dict = pickle.load(f)
        features= info_dict['features']
        labels = info_dict['labels']
        print('features: ', len(features))
        svclassifier_r.fit(features, labels)
        #
        fold_ckcpoint = make_ckcpoint(checkpoint=checkpoint, fold_idx=fold_idx, C=C)
        output_model_fold = join(fold_ckcpoint, 'model.pkl')
        with open(output_model_fold, 'wb') as f:
            pickle.dump(svclassifier_r, f)
        # test
        acc_fold = eval_visual(data=feature_fold_test, model_file=output_model_fold)
        os.rename(src=fold_ckcpoint, dst=join("/".join(fold_ckcpoint.split("/")[:-1]), '{}_fold_{}'.format(acc_fold, fold_idx)))
        break

def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data', dest='input',default='',
                        help='Path to pkl data file.')
    parser.add_argument('-o', '--model', dest='output', help='path to save model.',
                        default='./output')
    args = parser.parse_args()
    return args


if __name__ == "__name__":
    args_in = parse_args()
    train_visual(args_in.data,args_in.model)


