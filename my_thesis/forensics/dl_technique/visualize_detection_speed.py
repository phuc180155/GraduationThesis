#################### VISUALIZE INFERENCE TIME ####################
import numpy as np
import random
from module.train_torch import calculate_metric
from metrics.metric import calculate_cls_metrics
from torchsummary import summary
import torch
import time
from dataloader.gen_dataloader import *
import torch.nn as nn
import os
from loss.contrastive_loss import ContrastiveLoss
from torch.autograd import Variable
from model.cnn.capsule_net.model import VggExtractor, CapsuleNet
from loss.capsule_loss import CapsuleLoss
from sklearn import metrics
from tqdm import tqdm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def eval_kfold_image_stream(model ,dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0 ):
    loss = 0
    mac_accuracy = 0
    model.eval()
    y_label = []
    y_pred_label = []
    begin = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Push to device
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            # Forward network
            logps = model.forward(inputs)
            if len(logps.shape) == 2:
                logps = logps.squeeze(dim=1)

            # Loss in a batch
            batch_loss = criterion(logps, labels)
            # Cumulate into running val loss
            loss += batch_loss.item()

            # Find accuracy
            equals = (labels == (logps > 0.5))
            mac_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            
    assert len(y_label) == len(y_pred_label), "Bug"
    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    processing_time = time.time() - begin
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros, processing_time

def eval_kfold_capsulenet(capnet, vgg_ext, dataloader, device, capsule_loss, adj_brightness=1.0, adj_contrast=1.0 ):
    capnet.eval()
    vgg_ext.eval()

    y_label = []
    y_pred = []
    y_pred_label = []
    loss = 0
    mac_accuracy = 0
    begin = time.time()
    
    for inputs, labels in dataloader:
        labels[labels > 1] = 1
        img_label = labels.numpy().astype(np.float)
        inputs, labels = inputs.to(device), labels.to(device)

        input_v = Variable(inputs)
        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=False)

        loss_dis = capsule_loss(classes, Variable(labels, requires_grad=False))
        loss_dis_data = loss_dis.item()
        output_dis = class_.data.cpu().numpy()

        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        loss += loss_dis_data
        y_label.extend(img_label)
        y_pred.extend(output_dis)
        y_pred_label.extend(output_pred)
        mac_accuracy += metrics.accuracy_score(img_label, output_pred)
        
    mac_accuracy /= len(dataloader)
    loss /= len(dataloader)
    assert len(y_label) == len(y_pred_label), "Bug"
    ######## Calculate metrics:
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    processing_time = time.time() - begin
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros, processing_time

def eval_kfold_dual_stream(model, dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0):
    """ Evaluate model with dataloader

    Args:
        model (_type_): model weight
        dataloader (_type_): dataloader of [test or val]
        device (_type_): [gpu or cpu]
        criterion (_type_): loss module
        adj_brightness (float, optional): adjust brightness. Defaults to 1.0.
        adj_contrast (float, optional): adjust contrast. Defaults to 1.0.

    Returns:
        eval_loss, macro accuracy, micro accuracy, (precision/recall/f1-score) of (real class, fake class, micro average, macro average): metrics
    """
    loss = 0
    mac_accuracy = 0
    model.eval()
    # Find other metrics
    y_label = []
    y_pred_label = []
    begin = time.time()
    with torch.no_grad():
        for inputs, fft_imgs, labels in tqdm(dataloader):
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            # Push to device
            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), labels.float().to(device)

            # Forward network
            logps = model.forward(inputs, fft_imgs)

            if len(logps.shape) == 0:
                logps = logps.unsqueeze(dim=0)
            if len(logps.shape) == 2:
                logps = logps.squeeze(dim=1)

            # Loss in a batch
            batch_loss = criterion(logps, labels)
            # Cumulate into running val loss
            loss += batch_loss.item()

            # Find accuracy
            equals = (labels == (logps > 0.5))
            mac_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)

    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    processing_time = time.time() - begin
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros, processing_time

def eval_kfold_pairwise_image_stream_twooutput(model, weight_importance, dataloader, device, ce_loss, contrastive_loss, adj_brightness=1.0, adj_contrast=1.0):
    """ Evaluate model with dataloader

    Args:
        model (_type_): model weight
        dataloader (_type_): dataloader of [test or val]
        device (_type_): [gpu or cpu]
        criterion (_type_): loss module
        adj_brightness (float, optional): adjust brightness. Defaults to 1.0.
        adj_contrast (float, optional): adjust contrast. Defaults to 1.0.

    Returns:
        eval_loss, macro accuracy, micro accuracy, (precision/recall/f1-score) of (real class, fake class, micro average, macro average): metrics
    """
    contrastive_loss_ = 0
    ce_loss_ = 0
    total_loss_ = 0
    mac_accuracy = 0
    model.eval()
    # Find other metrics
    y_label = []
    y_pred_label = []
    begin = time.time()
    with torch.no_grad():
        for inputs0, fft_imgs0, labels0, inputs1, fft_imgs1, labels1, labels_contrastive in tqdm(dataloader):
            # Push to device
            y_label.extend(labels0.cpu().numpy().astype(np.float64))
            inputs0, fft_imgs0, labels0 = inputs0.float().to(device), fft_imgs0.float().to(device), labels0.long().to(device)
            inputs1, fft_imgs1, labels1 = inputs1.float().to(device), fft_imgs1.float().to(device), labels1.long().to(device)
            labels_contrastive = labels_contrastive.float().to(device)
            
            # Forward
            embedding_0, logps0, embedding_1, logps1  = model.forward(inputs0, fft_imgs0, inputs1, fft_imgs1)     # Shape (32, 1)

            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print("logps0: ", logps0.shape, '          ===== ', logps0)
            # print("labels0: ", labels0.shape, '          ===== ', labels0)
            # Find mean loss

            # print("logps0: ", logps0.shape, '          ===== ', logps0)
            # print("labels0: ", labels0.shape, '          ===== ', labels0)
            sys.stdout = sys.__stdout__
            celoss_0 = ce_loss(logps0, labels0)
            # bceloss_1 = ce_loss(logps1, labels1)
            contrastiveloss = contrastive_loss(embedding_0, embedding_1, labels_contrastive)
            batch_loss = weight_importance * celoss_0 + contrastiveloss

            # Cumulate into running val loss
            contrastive_loss_ += contrastiveloss.item()
            ce_loss_ += celoss_0.item()
            total_loss_ += batch_loss.item()

            # Find accuracy
            values, preds = torch.max(logps0, dim=1)
            mean_acc = torch.mean(torch.tensor(labels0.data == preds, dtype=torch.float32).cpu().detach()).item()
            mac_accuracy += mean_acc
            #
            pred_label = preds.cpu().detach().numpy()
            y_pred_label.extend(pred_label)

    ######## Calculate metrics:
    contrastive_loss_ /= len(dataloader)
    ce_loss_ /= len(dataloader)
    total_loss_ /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    processing_time = time.time() - begin
    # calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=True)
    return contrastive_loss_, ce_loss_, total_loss_, mac_accuracy, mic_accuracy, reals, fakes, micros, macros, processing_time

def get_model_size(model=None, dual=False, pairwise=False, capsule=False, vgg_ext=None, capnet=None):
    if capsule:
        summary(vgg_ext, (3, 128, 128), device='cuda')
        summary(capnet, (256, 16, 16), device='cuda')
        return

    if dual:
        if not pairwise:
            summary(model, [(3, 128, 128), (1, 128, 128)], device='cuda')
        else:
            summary(model, [(3, 128, 128), (1, 128, 128), (3, 128, 128), (1, 128, 128)], device='cuda')
    else:
        if not pairwise:
            summary(model, (3, 128, 128), device='cuda')
        else:
            summary(model, [(3, 128, 128), (3, 128, 128)], device='cuda')

def get_detection_speed_meso_xception(model=None, device=None, batchsize=32):
    test_dir = "/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/test"
    test_loader = generate_test_dataloader_single_cnn_stream_for_kfold(test_dir, 128, batchsize, 4)
    bce = nn.BCELoss().to(device)
    loss, acc, _, _, _ ,_, _, processing_time = eval_kfold_image_stream(model, test_loader, device, bce)
    print('loss: ', loss)
    print('acc: ', acc)
    print('processing time: ', processing_time)
    return processing_time

def get_detection_speed_capsulenet(capnet, vgg_ext, device=None, batchsize=32):
    time.sleep(5)
    test_dir = "/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/test"
    test_loader = generate_test_dataloader_single_cnn_stream_for_kfold(test_dir, 128, batchsize, 4)
    capsule_loss = CapsuleLoss().to(device)
    loss, acc, _, _, _ ,_, _, processing_time = eval_kfold_capsulenet(capnet, vgg_ext, test_loader, device, capsule_loss)
    print('loss: ', loss)
    print('acc: ', acc)
    print('processing time: ', processing_time)
    return processing_time

def get_detection_speed_dual_efficient(model, device=None, batchsize=32):
    time.sleep(5)
    test_dir = "/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/test"
    test_loader = generate_test_dataloader_dual_cnn_stream_for_kfold(test_dir, 128, batchsize, 4)
    bce = nn.BCELoss().to(device)
    loss, acc, _, _, _ ,_, _, processing_time = eval_kfold_dual_stream(model, test_loader, device, bce)
    print('loss: ', loss)
    print('acc: ', acc)
    print('processing time: ', processing_time)
    return processing_time

def get_detection_speed_pairwise_dual_dab_cnn_multivit(model, device=None, batchsize=32, weight_importance=1., margin=0.4):
    time.sleep(5)
    test_dir = "/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/test"
    test_loader = generate_test_dataloader_dual_cnn_stream_for_kfold_pairwise(test_dir, 128, batchsize, 4)
    ce_loss = nn.CrossEntropyLoss().to(device)
    contrastive_loss = ContrastiveLoss(device=device, margin=margin).to(device)
    contrastive_loss, loss, total_loss, acc, _, _, _ ,_, _, processing_time = eval_kfold_pairwise_image_stream_twooutput(model, weight_importance, test_loader, device, ce_loss, contrastive_loss)
    print('loss: ', loss)
    print('acc: ', acc)
    print('processing time: ', processing_time)
    return processing_time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_capsule = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/checkpoint/extend_data/deepfake/kfold_capsule/lr0.01_batch16_esnone_nf5_trick1_beta0.9_dropout0.05_seed0_drmlp0.0_aug0/(1.9595_0.8952_0.9476)_fold_1/step/best_val_loss_13400_1.959492.pt"
checkpoint_meso = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/checkpoint/extend_data/deepfake/kfold_meso4/lr0.0002_batch16_esnone_lossbce_nf5_trick1_seed0_drmlp0.0_aug0/(0.0073_0.9855_0.9941)_fold_1/step/best_val_loss_20000_0.007320.pt"
checkpoint_xception = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/checkpoint/extend_data/deepfake/kfold_xception/lr0.0002_batch16_esnone_lossbce_nf5_trick1_pre1_seed0_drmlp0.0_aug0/(0.0000_1.0000)_fold_1/step/best_val_loss_12600_0.000044.pt"
checkpoint_dual_efficient = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/checkpoint/tmp/dual_efficient/lr0.0002_batch32_esnone_lossbce_nf5_trick1_pre1_seed0_drmlp0.0_aug0/best_test_acc_300_0.990800.pt"
checkpoint_proposal = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/checkpoint/tmp/pairwise_dual_dab_cnn_multiviit/lr0.0002-0_b32_c1_esnone_lbce_nf5trick0_retmlp_hidden_im1.0mar0.4_vca-ifadd-0.8_KNN0_d512md1024h4d3_bbefficient_netpre1_fatb11_pres1-2-4_res1_gres0.2_0.2_0.2_sh0_nrmbatchnorm_qkv1_prj1_actselunone_topk0.35_dabmca_dabinone_dabb8_9_10_sd0_dr0.3aug0/best_test_acc_10200_0.999700.pt"
dual=True
pairwise=True
batchsize=32

################### MESONET4 ###################
# from model.cnn.mesonet4.model import Meso4
# model_ = Meso4(num_classes=1, image_size=128)
# model_ = model_.to(device)
# model_.load_state_dict(torch.load(checkpoint_meso))
# get_model_size(model=model_, dual=dual, pairwise=pairwise)
# time_total = 0
# for _ in range(5):
#     time_total += get_detection_speed_meso_xception(model=model_, device=device, batchsize=batchsize)
# print(time_total/5)

################### XCEPTION ###################
# from model.cnn.xception_net.model import xception
# model_ = xception(pretrained=False)
# model_ = model_.to(device)
# model_.load_state_dict(torch.load(checkpoint_xception))
# get_model_size(model=model_, dual=dual, pairwise=pairwise)
# time_total = 0
# for _ in range(5):
#     time_total += get_detection_speed_meso_xception(model=model_, device=device, batchsize=batchsize)
# print(time_total/5)

################### CAPSULENET ###################
# vgg_ext = VggExtractor().to(device)
# capnet = CapsuleNet(num_class=2, device=device).to(device)
# capnet.load_state_dict(torch.load(checkpoint_capsule))
# get_model_size(model=None, dual=dual, pairwise=pairwise, capsule=True, vgg_ext=vgg_ext, capnet=capnet)
# time_total = 0
# for _ in range(5):
#     time_total += get_detection_speed_capsulenet(capnet, vgg_ext, device=device, batchsize=batchsize)
# print(time_total/5)

################### DUAL EFFICIENT ###################
# from model.cnn.dual_cnn.dual_efficient import DualEfficient
# model_ = DualEfficient(pretrained=False)
# model_ = model_.to(device)
# model_.load_state_dict(torch.load(checkpoint_dual_efficient))
# # get_model_size(model=model_, dual=dual, pairwise=pairwise)
# time_total = 0
# for _ in range(5):
#     time_total += get_detection_speed_dual_efficient(model=model_, device=device, batchsize=batchsize)
# print(time_total/5)

################### PROPOSAL ###################
from model.vision_transformer.dual_cnn_vit.pairwise_dual_dab_cnn_cmultivit_twoout import PairwiseDualDabCNNCMultiViT2

model_ = PairwiseDualDabCNNCMultiViT2(image_size=128, num_classes=2, \
        dim=512, depth=3, heads=4, mlp_dim=1024, dim_head=64, dropout=0.,\
        backbone='efficient_net', pretrained=False,unfreeze_blocks=-1,\
        normalize_ifft='batchnorm',\
        qkv_embed=True, prj_out=True, act='selu',\
        patch_reso='1-2-4', gammaagg_reso='0.2_0.2_0.2',\
        fusca_version='ca-ifadd-0.8',\
        features_at_block='11',\
        dropout_in_mlp=0.3, residual=1, transformer_shareweight=0,\
        act_dab='none', topk_channels=0.35, dab_modules='ca', dabifft_normalize='none', dab_blocks='8_9_10',\
        embedding_return='mlp_hidden', useKNN=0)
model_ = model_.to(device)
model_.load_state_dict(torch.load(checkpoint_proposal))
# get_model_size(model=model_, dual=dual, pairwise=pairwise)
time_total = 0
for _ in range(5):
    time_total += get_detection_speed_pairwise_dual_dab_cnn_multivit(model=model_, device=device, batchsize=batchsize, weight_importance=1.0, margin=0.4)
print(time_total/5)

