import os, pickle, argparse
from headpose_forensic.utils.proc_vid import parse_vid, parse_img
from headpose_forensic.utils.face_proc import FaceProc
from PIL import ImageEnhance, Image
import random
from headpose_forensic.utils.head_pose_proc import PoseEstimator
from tqdm import tqdm

def main1(input_real,input_fake, main1_file,number_iter):
    face_inst = FaceProc()
    info_dict = {}
    video_dir_dict = {}
    video_dir_dict['real'] = input_real
    video_dir_dict['fake'] = input_fake
    limited_samples = True if number_iter != -1 else False
    number_iter_real = len(input_real) if number_iter == -1 else number_iter
    number_iter_fake = len(input_fake) if number_iter == -1 else number_iter
    for tag in video_dir_dict:
        cont = 0
        vid_list = video_dir_dict[tag]
        random.shuffle(vid_list)
        number_iter = number_iter_real if tag == 'real' else number_iter_fake
        for vid_name in tqdm(vid_list):
            vid_path = vid_name
            #             print( 'processing video: ', vid_path)
            info = {'height': [], 'width': [], 'label': []}
            img, width, height = parse_img(vid_path)
            ##
            #             img = img.astype("uint8")
            #             contrast = ImageEnhance.Contrast(Image.fromarray(img))
            #             img = contrast.enhance(2.0)
            #             brightness = ImageEnhance.Brightness(img)
            #             img = brightness.enhance(1.0)
            #             img = np.array(img,dtype='uint8')
            ##
            info['label'] = tag
            info['height'] = height
            info['width'] = width
            # mark_list_all = []
            landmarks = face_inst.get_landmarks(img)
            # mark_list_all.append(landmarks)
            info['landmarks'] = landmarks
            info_dict[vid_path] = info
            cont += 1
            #             print(cont)

    with open(main1_file, 'wb') as f:
        pickle.dump(info_dict, f)

def main2(main1_file, main2_file):
    with open(main1_file, 'rb') as f:
        vids_info = pickle.load(f)

    markID_c = '18-36,49,55'
    markID_a = '1-36,49,55'
    save_pose_file = main2_file

    for key, value in vids_info.items():
#         print(key)
        # Load 2d landmarks
        landmark_2d = value['landmarks']
        height = value['height']
        width = value['width']
        pose_estimate = PoseEstimator([height, width])

        R_c_list, R_a_list, t_c_list, t_a_list = [], [], [], []
        R_c_matrix_list, R_a_matrix_list = [], []
        # for landmark_2d_cur in landmark_2d:
        landmark_2d_cur = landmark_2d
        # landmark_2d_cur = landmark_2d[i]
        if landmark_2d_cur is not None:
            R_c, t_c = pose_estimate.solve_single_pose(landmark_2d_cur, markID_c)
            R_a, t_a = pose_estimate.solve_single_pose(landmark_2d_cur, markID_a)

            R_c_matrix = pose_estimate.Rodrigues_convert(R_c)
            R_a_matrix = pose_estimate.Rodrigues_convert(R_a)


            R_c_list.append(R_c)
            R_a_list.append(R_a)

            t_c_list.append(t_c)
            t_a_list.append(t_a)

            R_c_matrix_list.append(R_c_matrix)
            R_a_matrix_list.append(R_a_matrix)

        value['R_c_vec'] = R_c_list
        value['R_c_mat'] = R_c_matrix_list
        value['t_c'] = t_c_list

        value['R_a_vec'] = R_a_list
        value['R_a_mat'] = R_a_matrix_list
        value['t_a'] = t_a_list

    # Save to pkl file
    with open(save_pose_file, 'wb') as f:
        pickle.dump(vids_info, f)
    print('Done!')

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
    if len(files) == n_folds + 1:
        feature_fold_train = [feature_ckcpoint + '/train_fold{}.pkl'.format(fold_idx) for fold_idx in range(n_folds)]
        feature_fold_test = feature_ckcpoint + '/test.pkl'
        print("Extracted: ", time() - begin)
        return feature_fold_train, feature_fold_test
        
    for fold_idx in range(n_folds):
        print("          *****: FOLD {} ".format(fold_idx))
        trainset, valset = kfold.get_fold(fold_idx=fold_idx)
        real_path, fake_path = split_real_fake(train_paths=trainset)
        output_fold = feature_ckcpoint + '/train_fold{}.pkl'.format(fold_idx)
        main1(real_path, fake_path, "headpose_forensic/output_features/temp_train_{}_{}.p".format(datasetname, fold_idx), -1)
        main2("headpose_forensic/output_features/temp_train_{}_{}.p".format(datasetname, fold_idx), output_fold)
        feature_fold_train.append(output_fold)
    # test
    testset = get_test_path(test_dir=test_dir)
    real_path, fake_path = split_real_fake(train_paths=testset)
    output_fold = feature_ckcpoint + '/test.pkl'
    main1(real_path, fake_path, "headpose_forensic/output_features/temp_test_{}.p".format(datasetname), -1)
    main2("headpose_forensic/output_features/temp_test_{}.p".format(datasetname), output_fold)
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
    main1(args_in.input_real,args_in.input_fake,"temp.p", int(args_in.number_iter))
    main2("temp.p", args_in.output)
