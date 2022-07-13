import os
from torch.utils.data import DataLoader

import torch

import torch.nn as nn
from torch import optim
import glob
from torchvision import transforms, datasets, models
from pytorch_model.pairwise.dense_block import DenseBlock,TransitionBlock,BasicBlock,BottleneckBlock

class cffn(nn.Module):
    def __init__(self,image_size = 256):
        super(cffn, self).__init__()
        growth_rate = 24
        dropRate = 0.0
        self.image_size = image_size
        reduction = 0.5
        self.conv1 = nn.Conv2d(3, 48, kernel_size=7, stride=4,
                               padding=2, bias=False)   # 48 x 64 x 64
        self.block1 = DenseBlock(2, 48, growth_rate, BasicBlock, dropRate)

        self.trans1 = TransitionBlock(48+2*24, 60, dropRate=dropRate)

        self.block2 = DenseBlock(3, 60, growth_rate, BasicBlock, dropRate)# 60x32x32
        self.trans2 = TransitionBlock(60+3*24, 78, dropRate=dropRate)

        self.block3 = DenseBlock(4, 78, growth_rate, BasicBlock, dropRate)# 78x16x16
        self.trans3 = TransitionBlock(78+4*24, 126, dropRate=dropRate)

        self.block4 = DenseBlock(2, 126, growth_rate, BasicBlock, dropRate)# 126x8x8
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((48+2*24)*(int(self.image_size/4))**2+(60+3*24)*(int(self.image_size/8))**2+(78+4*24)*(int(self.image_size/16))**2+(126+2*24)*(int(self.image_size/32))**2, 128)

    def forward(self, input):
        x = self.conv1(input)
        x1 = self.block1(x)
        x = self.trans1(x1)
        x2 = self.block2(x)
        x = self.trans2(x2)
        x3 = self.block3(x)
        x = self.trans3(x3)
        xc = self.block4(x)
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)
        x4 = self.flatten(xc)
        x = torch.cat([x1,x2,x3,x4],1)
        x = self.fc(x)
        return x,xc
class classify(nn.Module):
    def __init__(self):
        super(classify, self).__init__()
        self.conv1 = nn.Conv2d((126+2*24), 2, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
class Pairwise(nn.Module):
    def __init__(self,image_size):
        super(Pairwise, self).__init__()
        self.cffn = cffn(image_size)

    def forward_once(self, x):
        output = self.cffn(x)

        return output

    def forward(self, input1, input2):
        output1,_ = self.forward_once(input1)
        output2,_ = self.forward_once(input2)

        return output1, output2

class ClassifyFull(nn.Module):
    def __init__(self,image_size):
        super(ClassifyFull, self).__init__()
        self.cffn = cffn(image_size)
        for param in self.cffn.parameters():
            param.requires_grad = False
        self.classify = classify()


    def forward(self, input):
        x,x_c = self.cffn(input)
        x = self.classify(x_c)
        return x




if __name__ == "__main__":
    from pytorch_model.pairwise.contrastive_loss import ContrastiveLoss
    from pytorch_model.pairwise.data_generate import get_generate_pairwise

    image_size = 128

    train_set = "../../../../extract_raw_img"
    val_set = "../../../../extract_raw_img"
    batch_size = 2
    train_number_epochs = 1
    checkpoint = "checkpoint"
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    model = Pairwise(image_size).to(device)
    # model.summary()
    model_clss = ClassifyFull(image_size)
    # model_clss.cffn.load_state_dict(torch.load(os.path.join(checkpoint, 'pairwise_0.pt')))
    # torch.save(model_clss.state_dict(), os.path.join(checkpoint, 'cls_sss.pt'))

    criterion = ContrastiveLoss(device)
    criterion2 = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataloader_train = get_generate_pairwise(train_set, image_size, batch_size, 1)
    import time
    from tqdm import tqdm
    iteration_number = 0
    for epoch in range(0, train_number_epochs):
        for i, data in enumerate(dataloader_train, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device),label.to(device)
            print(i)
            # print(y1,"   ",y2,"   ",label)
            # continue
            #         img0, img1 , label = img0, img1 , label
            # print(img0.size())
            optimizer.zero_grad()
            output1, output2= model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            print(loss_contrastive)

            optimizer.step()
            if i % 2 == 0:
                print("Epoch number {}\n iteration_number {} Current loss {}\n".format(epoch, iteration_number,
                                                                                       loss_contrastive.item()))
                iteration_number += 2
                # counter.append(iteration_number)
                # loss_history.append(loss_contrastive.item())
                if i % 2 == 0:
                    torch.save(model.cffn.state_dict(), os.path.join(checkpoint, 'pairwise_%d.pt' % epoch))
    torch.save(model.cffn.state_dict(), os.path.join(checkpoint, 'pairwise_100.pt' ))

