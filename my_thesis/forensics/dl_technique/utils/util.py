from logging import raiseExceptions
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
from os.path import join

def save_list_to_file(saved_list, file_name: str, overwrite=False):
    if isinstance(saved_list, np.ndarray):
        saved_list = saved_list.tolist()
    assert isinstance(saved_list, list), "Type of iteration should be a list!"
    with open('result/' + file_name, 'w' if overwrite else 'a') as f:
        f.write('=====================================\n')
        f.write(','.join([str(int(ele)) for ele in saved_list]))
        f.write('\n*************************************')


def is_refined_model(model_name: str):
    BASELINES = ['xception', 'dual_efficient', 'capsulenet', 'srm_2_stream', 'meso4', 'vit', 'swin_vit', 'm2tr']
    PROPOSALS = ['dual_attn_cnn', 'efficient_vit', \
                 'origin_dual_efficient_vit', 'origin_dual_efficient_vit_remove_ifft', \
                 'dual_cma_cnn_vit', 'dual_cnn_vit',  'pairwise_dual_cnn_vit', \
                 'dual_cnn_feedforward_vit', 'pairwise_dual_cnn_feedforward_vit', \
                 'dual_cma_cnn_attn', 'dual_cnn_cma_transformer', \
                 'dual_patch_cnn_cma_vit', 'pairwise_dual_patch_cnn_cma_vit', \
                 'dual_cnn_vit_test']
    if model_name in BASELINES:
        return False
    if model_name in PROPOSALS:
        return True
    raise Exception("Model not exists.")
            

def get_test_metric_by_step(step_ckcpoint: str, steps: int, pairwise=True, return_bestvalloss=True):
    #####
    step_best_val_bceloss = steps

    #####
    metrics = {
        'best_test_acc': {
            'acc': 0.,
            'pre': 0.,
            'rec': 0.,
            'f1': 0.
        },
        'best_val_bceloss': {
            'acc': 0.,
            'pre': 0.,
            'rec': 0.,
            'f1': 0.
        }
    }
    if pairwise:
        metrics.update({
            'best_val_totalloss': {
                'acc': 0.,
                'pre': 0.,
                'rec': 0.,
                'f1': 0.
            }
        })
    #####
    test_df = pd.read_csv(join(step_ckcpoint, 'result_test.csv'))
    val_df = pd.read_csv(join(step_ckcpoint, 'result_val.csv'))
    best_valtotalloss = 0.
    
    if not pairwise:
        ##### Best bceloss:
        idx_best_val_bcelosses = val_df.index[test_df["step"] == step_best_val_bceloss].tolist()

        idx_best_val_bceloss = idx_best_val_bcelosses[0]
        best_valtotalloss = val_df[" Val loss"].iloc[idx_best_val_bceloss]
        step_best_val_bceloss = val_df["step"].iloc[idx_best_val_bceloss]
        assert step_best_val_bceloss == test_df["step"].iloc[idx_best_val_bceloss], "Two step must be equal! {} - {}".format(step_best_val_bceloss, test_df["step"].iloc[idx_best_val_bceloss])
        acc_best_val_bceloss = test_df[" Test accuracy"].iloc[idx_best_val_bceloss]
        pre_best_val_bceloss, rec_best_val_bceloss, f1_best_val_bceloss = test_df[" Test macro pre"].iloc[idx_best_val_bceloss], test_df[" Test macro rec"].iloc[idx_best_val_bceloss], test_df[" Test macro F1-Score"].iloc[idx_best_val_bceloss]
        metrics['best_val_bceloss'] = {
            'acc': acc_best_val_bceloss, 
            'pre': pre_best_val_bceloss,
            'rec': rec_best_val_bceloss,
            'f1': f1_best_val_bceloss
        }
        # print("         ** Best val bceloss: at: ", step_best_val_bceloss, "  - acc: ", acc_best_val_bceloss, " - pre: ", pre_best_val_bceloss, " - rec: ", rec_best_val_bceloss, " - F1: ", f1_best_val_bceloss)
    else:
        ##### Best bceloss:
        idx_best_val_bcelosses = val_df.index[test_df["step"] == step_best_val_bceloss].tolist()
        if len(idx_best_val_bcelosses) != 1:
            print("Error in bce loss")
            exit(0)
            return
        idx_best_val_bceloss = idx_best_val_bcelosses[0]
        best_valtotalloss = val_df[" Val bce loss"].iloc[idx_best_val_bceloss]
        step_best_val_bceloss = val_df["step"].iloc[idx_best_val_bceloss]
        assert step_best_val_bceloss == test_df["step"].iloc[idx_best_val_bceloss], "Two step must be equal! {} - {}".format(step_best_val_bceloss, test_df["step"].iloc[idx_best_val_bceloss])
        acc_best_val_bceloss = test_df[" Test accuracy"].iloc[idx_best_val_bceloss]
        pre_best_val_bceloss, rec_best_val_bceloss, f1_best_val_bceloss = test_df[" Test macro pre"].iloc[idx_best_val_bceloss], test_df[" Test macro rec"].iloc[idx_best_val_bceloss], test_df[" Test macro F1-Score"].iloc[idx_best_val_bceloss]
        metrics['best_val_bceloss'] = {
            'acc': acc_best_val_bceloss, 
            'pre': pre_best_val_bceloss,
            'rec': rec_best_val_bceloss,
            'f1': f1_best_val_bceloss
        }    
        # print("         ** Best val bceloss: at: ", step_best_val_bceloss, "  - acc: ", acc_best_val_bceloss, " - pre: ", pre_best_val_bceloss, " - rec: ", rec_best_val_bceloss, " - F1: ", f1_best_val_bceloss)

    return metrics, best_valtotalloss if return_bestvalloss else metrics 


def get_test_metric_by_step2(step_ckcpoint: str, steps: int, pairwise=True, return_bestvalloss=True):
    #####
    step_best_test_acc = steps

    #####
    metrics = {
        'best_test_acc': {
            'acc': 0.,
            'pre': 0.,
            'rec': 0.,
            'f1': 0.
        },
        'best_val_bceloss': {
            'acc': 0.,
            'pre': 0.,
            'rec': 0.,
            'f1': 0.
        }
    }
    if pairwise:
        metrics.update({
            'best_val_totalloss': {
                'acc': 0.,
                'pre': 0.,
                'rec': 0.,
                'f1': 0.
            }
        })
    
    #####
    test_df = pd.read_csv(join(step_ckcpoint, 'result_test.csv'))
    val_df = pd.read_csv(join(step_ckcpoint, 'result_val.csv'))
    best_valtotalloss = 0.
    ##### Best test acc:
    idx_best_test_accs = test_df.index[test_df["step"] == step_best_test_acc].tolist()
    if len(idx_best_test_accs) != 1:
        print("Error in best test acc")
        exit(0)
        return
    idx_best_test_acc = idx_best_test_accs[0]
    step_best_test_acc = test_df["step"].iloc[idx_best_test_acc]
    acc_best_test_acc = test_df[" Test accuracy"].iloc[idx_best_test_acc]
    pre_best_test_acc, rec_best_test_acc, f1_best_test_acc = test_df[" Test macro pre"].iloc[idx_best_test_acc], test_df[" Test macro rec"].iloc[idx_best_test_acc], test_df[" Test macro F1-Score"].iloc[idx_best_test_acc]
    metrics['best_test_acc'] = {
        'acc': acc_best_test_acc, 
        'pre': pre_best_test_acc,
        'rec': rec_best_test_acc,
        'f1': f1_best_test_acc
    }
    # print("         ** Best test acc: at: ", step_best_test_acc, "  - acc: ", acc_best_test_acc, " - pre: ", pre_best_test_acc, " - rec: ", rec_best_test_acc, " - F1: ", f1_best_test_acc)

    return metrics, best_valtotalloss if return_bestvalloss else metrics 
        