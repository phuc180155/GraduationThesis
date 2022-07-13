import argparse
from sklearn.svm import SVC
import pickle
from spectrum.preprocess_data import extract_features_kfold
# load feature file
from spectrum.eval_spectrum import eval_spectrum
import os
import os.path as osp
import sys
from os.path import join
from util import make_ckcpoint

def train_spectrum(model_name: str, train_dir: str, val_dir: str, test_dir: str, checkpoint: str, n_folds: int, use_trick: int, C: float):
    feature_fold_train, feature_fold_test = extract_features_kfold(model_name=model_name, n_folds=n_folds, use_trick=use_trick, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)
    #
    for fold_idx in range(n_folds):
        ## train
        pkl_file = open(feature_fold_train[fold_idx], 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        X = data["data"]
        y = data["label"]
        svclassifier_r = SVC(C=C, kernel='rbf', gamma=0.86)
        svclassifier_r.fit(X, y)
        #
        fold_ckcpoint = make_ckcpoint(checkpoint=checkpoint, fold_idx=fold_idx, C=C)
        output_model_fold = join(fold_ckcpoint, 'model.pkl')
        with open(output_model_fold, 'wb') as f:
            pickle.dump(svclassifier_r, f)
        # test
        acc_fold = eval_spectrum(data=feature_fold_test, model_file=output_model_fold)
        os.rename(src=fold_ckcpoint, dst=join("/".join(fold_ckcpoint.split("/")[:-1]), '{}_fold_{}'.format(acc_fold, fold_idx)))