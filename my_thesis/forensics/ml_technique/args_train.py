import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import torch
import torch.nn as nn
import argparse
import json
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_dir', type=str, default="", help="path to train data")
    parser.add_argument('--val_dir', type=str, default="", help="path to validation data")
    parser.add_argument('--test_dir', type=str, default="", help="path to test data")
    parser.add_argument('--checkpoint',default = None,required=False, help='path to checkpoint ')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--use_trick', type=int, default=1)
    parser.add_argument('--C', type=float, default=1)

    sub_parser = parser.add_subparsers(dest="model", help="")
    parser_spectrum = sub_parser.add_parser('spectrum', help='spectrum')
    parser_visual_artifact = sub_parser.add_parser('visual_artifact', help='visual_artifact')
    parser_headpose = sub_parser.add_parser('headpose_forensic', help='visual_artifact')
    return parser.parse_args()

import torch
import os
import random
import torch.backends.cudnn as cudnn
import numpy as np
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 
torch.set_num_threads(2)

def config_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    model = args.model
    
    # Save args to text:
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    config_seed(seed=args.seed)
        
    ################# TRAIN #######################n
    if model == "spectrum":
        from spectrum.train_spectrum import train_spectrum
        train_spectrum(model_name=model, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, checkpoint=args.checkpoint, n_folds=args.n_folds, use_trick=args.use_trick, C=args.C)
    elif model == "visual_artifact":
        from visual_artifact.train_visual import train_visual
        train_visual(model_name=model, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, checkpoint=args.checkpoint, n_folds=args.n_folds, use_trick=args.use_trick, C=args.C)
    elif model == "headpose_forensic":
        from headpose_forensic.train_headpose import train_headpose
        train_headpose(model_name=model, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, checkpoint=args.checkpoint, n_folds=args.n_folds, use_trick=args.use_trick, C=args.C)