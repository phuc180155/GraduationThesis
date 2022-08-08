import torch.nn as nn
from torch import einsum
import torch
from torchsummary import summary
from einops import rearrange

import sys
from model.backbone.efficient_net.model import EfficientNet

import re
import torch.nn.functional as F

import re, math
from model.vision_transformer.vit.vit import ViT, Transformer
from model.vision_transformer.vit.kvit import kNNTransformer
from model.vision_transformer.cnn_vit.efficient_vit import EfficientViT
from pytorchcv.model_provider import get_model



class CMultiscaleViT(nn.Module):
    def __init__(self, in_channels=112, in_size=8, patch_reso='1-2-4-8', gamma_reso='0.8_0.4_0.2_0.1', residual=True,\
                useKNN=True, depth=6, heads=8, dim=1024, mlp_dim=2048, dim_head=64, dropout=0.15, share_weight=True):
        super(CMultiscaleViT, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout_value = dropout

        self.residual = residual
        self.patch_reso = patch_reso
        self.gamma_reso = gamma_reso

        self.patch_size = list(map(int, patch_reso.split('-')))
        self.gamma_reso = list(map(float, gamma_reso.split('_')))
        self.gamma = []
        self.g0 = nn.Parameter(torch.ones(1))
        self.g1 = nn.Parameter(torch.ones(1))
        self.g2 = nn.Parameter(torch.ones(1))
        self.g3 = nn.Parameter(torch.ones(1))
        cnt = 0
        if residual:
            for g in self.gamma_reso:
                if g != 0:
                    self.gamma.append(g)
                else:
                    if cnt == 0:
                        self.gamma.append(self.g0)
                    if cnt == 1:
                        self.gamma.append(self.g1)
                    if cnt == 2:
                        self.gamma.append(self.g2)
                    if cnt == 3:
                        self.gamma.append(self.g3)
                    cnt += 1

        self.num_patches = [int((in_size // p)** 2) for p in self.patch_size]
        self.n_chunks = len(self.patch_size)
        with torch.no_grad():
            test_inp = torch.ones(4, in_channels, in_size, in_size)
            test_chunks = torch.chunk(test_inp, chunks=self.n_chunks, dim=1)
            self.chunk_channels = [chunk.shape[1] for chunk in test_chunks]
        self.patch_dim = [int(self.chunk_channels[i] * (self.patch_size[i] ** 2)) for i in range(self.n_chunks)]

        ############################# VIT #########################################
        # Giảm chiều vector sau concat 2*patch_dim về D:
        self.embedding = nn.ModuleList([])
        for p_dim in self.patch_dim:
            self.embedding.append(nn.Linear(p_dim, self.dim))
        # transformer:
        self.share_weight = share_weight
        if not share_weight:
            self.transformers = nn.ModuleList([])
            for _ in range(len(self.patch_size)):
                if useKNN == 0:
                    print("use vanilla attention")
                    self.transformers.append(Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value))
                elif useKNN > 0:
                    print("use KNN attention: topK ratio: ", useKNN)
                    self.transformers.append(kNNTransformer(self.dim, self.depth, self.heads, self.mlp_dim, self.dropout_value, useKNN))
                else:
                    print("error when use attention...")
        else:
            if useKNN == 0:
                print("use vanilla attention")
                self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value)
            elif useKNN > 0:
                print("use KNN attention: topK ratio: ", useKNN)
                self.transformer = kNNTransformer(self.dim, self.depth, self.heads, self.mlp_dim, self.dropout_value, useKNN)
            else:
                print("error when use attention...")
        
    def forward(self, features):
        outputs = []
        chunked_features = torch.chunk(features, self.n_chunks, dim=1)

        for i in range(self.n_chunks):
            # Flatten to vectors:
            # print("shape: ", chunked_features[i].shape)
            feature_vectors = self.flatten_to_vectors(feature=chunked_features[i], p_size=self.patch_size[i])      # B, num_patch, patch_dim
            
            # print("patchsize: ", self.patch_size[i])
            # print("     Vectors shape: ", feature_vectors.shape)

            # ViT:
            embed = self.embedding[i](feature_vectors)                # B, num_patch, dim
            if not self.share_weight:       
                output = self.transformers[i](embed)                # B, num_patch, dim
            else:
                output = self.transformer(embed)
            #
            if self.residual:
                output = embed + self.gamma[i] * output        # B, num_patch, dim
            # print("     output shape: ", output.shape)
            output = output.mean(dim = 1).squeeze(dim=1)          # B, 1, dim
            outputs.append(output)
        
        out = torch.cat(outputs, dim=1)
        # print("multi shape: ", out.shape)
        return out

    def flatten_to_vectors(self, feature=None, p_size=1):
        return rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p_size, p2=p_size)

    
    
class PairwiseCNNCMultiViT(nn.Module):
    def __init__(self, image_size=224, num_classes=1, \
                dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head=64, dropout=0.15,\
                backbone='xception_net', pretrained=True,unfreeze_blocks=-1,\
                patch_reso='1-2-4-8', gammaagg_reso='0.8_0.4_0.2_0.1',\
                features_at_block='10',\
                dropout_in_mlp=0.0, residual=True, transformer_shareweight=True, useKNN=0, \
                embedding_return='mlp_hidden', freq_stream=True):  

        super(PairwiseCNNCMultiViT, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.backbone = backbone
        self.embedding_return = embedding_return

        self.features_size = {
            'efficient_net': {
                '0': (16, 64, 64),
                '1': (24, 32, 32),
                '2': (24, 32, 32),
                '3': (40, 16, 16),
                '4': (40, 16, 16),
                '5': (80, 8, 8),
                '6': (80, 8, 8),
                '7': (80, 8, 8),
                '8': (112, 8, 8),
                '9': (112, 8, 8),
                '10': (112, 8, 8),
                '11': (192, 4, 4),
                '12': (192, 4, 4),
                '13': (192, 4, 4),
                '14': (192, 4, 4),
                '15': (320, 4, 4),
                'final': (1280, 4, 4)
            },
            'xception_net': {
                'final': (2048, 4, 4)
            }
        }
        self.out_ext_channels = self.features_size[backbone][features_at_block][0]
        self.out_ext_size = self.features_size[backbone][features_at_block][1]

        self.pretrained = pretrained
        self.features_at_block = features_at_block

        self.extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=(3 if freq_stream == 'rgb' else 1))   # efficient_net-b0, return shape (1280, 8, 8) or (1280, 7, 7)

        self.multi_transformer = CMultiscaleViT(in_channels=self.out_ext_channels, in_size=self.out_ext_size, patch_reso=patch_reso, gamma_reso=gammaagg_reso,\
                                          useKNN=useKNN, depth=depth, heads=heads, dim=dim, mlp_dim=mlp_dim, dim_head=dim_head, dropout=dropout, residual=residual, share_weight=transformer_shareweight)

        self.mlp_relu = nn.ReLU(inplace=True)
        self.mlp_head_hidden = nn.Linear(len(patch_reso.split('-')) * dim, mlp_dim)
        self.mlp_dropout = nn.Dropout(dropout_in_mlp)
        self.mlp_head_out = nn.Linear(mlp_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def get_feature_extractor(self, architecture="efficient_net", unfreeze_blocks=-1, pretrained=False, num_classes=1, in_channels=3):
        extractor = None
        if architecture == "efficient_net":
            extractor = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes,in_channels = in_channels, pretrained=bool(pretrained))
            if unfreeze_blocks != -1:
                # Freeze the first (num_blocks - 3) blocks and unfreeze the rest 
                for i in range(0, len(extractor._blocks)):
                    for index, param in enumerate(extractor._blocks[i].parameters()):
                        if i >= len(extractor._blocks) - unfreeze_blocks:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
        
        if architecture == 'xception_net':
            xception = get_model("xception", pretrained=bool(pretrained))
            extractor = nn.Sequential(*list(xception.children())[:-1])
            extractor[0].final_block.pool = nn.Identity()
            if in_channels != 3:
                extractor[0].init_block.conv1.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

            if unfreeze_blocks != -1:
                blocks = len(extractor[0].children())
                print("Number of blocks in xception: ", len(blocks))
                for i, block in enumerate(extractor[0].children()):
                    if i >= blocks - unfreeze_blocks:
                        for param in block.parameters():
                            param.requires_grad = True
                    else:
                        for param in block.parameters():
                            param.requires_grad = False
        print("Pretrained backbone: ", bool(pretrained))
        return extractor


    def extract_feature(self, imgs):
        if self.backbone == 'efficient_net':
            if self.features_at_block == 'final':
                features = self.extractor.extract_features(imgs)                 # shape (batchsize, 1280, 8, 8)
            else:
                features = self.extractor.extract_features_at_block(imgs, int(self.features_at_block))
        else:
            features = self.extractor(imgs)
        return features

    def forward_once(self, imgs):
        features = self.extract_feature(imgs)
        # print("Features shape: ", features.shape)

        ##### Forward to ViT
        e = None
        e1 = self.multi_transformer(features)     # B, number_of_patch * D

        x = self.mlp_dropout(e1)         # B, number_of_patch * D
        e2 = self.mlp_head_hidden(x)     # B, number_of_patch * D => B, mlp_dim
        x = self.mlp_relu(e2)
        x = self.mlp_dropout(x)
        e3 = self.mlp_head_out(x)        # B, mlp_dim => B, 1
        x = self.sigmoid(e3)
        if self.embedding_return == 'mlp_before':
            e = e1
        elif self.embedding_return == 'mlp_hidden':
            e = e2
        elif self.embedding_return == 'mlp_out':
            e = e3
        return e, x

    def forward(self, imgs0, imgs1):
        embedding_0, out_0 = self.forward_once(imgs0)
        embedding_1, out_1 = self.forward_once(imgs1)
        return embedding_0, out_0, embedding_1, out_1

from torchsummary import summary
if __name__ == '__main__':
    x = torch.ones(32, 3, 128, 128)
    model_ = PairwiseCNNCMultiViT(image_size=128, num_classes=1, \
                dim=512, depth=3, heads=4, mlp_dim=1024, dim_head=64, dropout=0.15,\
                backbone='efficient_net', pretrained=True,unfreeze_blocks=-1,\
                patch_reso='1-2-4', gammaagg_reso='0.2_0.2_0',\
                features_at_block='11',\
                dropout_in_mlp=0.0, residual=True, transformer_shareweight=True, useKNN=0, \
                embedding_return='mlp_hidden', freq_stream='rgb')

    e0, out0, e1, out1 = model_(x, x)
    print(e0.shape)
    print(out0.shape)
