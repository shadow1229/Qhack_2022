import os,sys,glob
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net_16q(nn.Module):
    def __init__(self):
        super(Net_16q, self).__init__()
        #done by quanv layer
        #self.conv_i1  = nn.Conv3d(3,   8, 1,padding=0)
        #self.conv1   = atr_res_12(8)
        #self.conv2   = atr_res_12(8)
        #self.maxpool1 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_i2  = nn.Conv3d(16,   64, 1,padding=0)
        self.conv3   = atr_res_12(64)
        self.conv4   = atr_res_12(64)
        self.maxpool2 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv5   = atr_res_12(64)
        self.conv6   = atr_res_12(64)
        self.maxpool3 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv7   = atr_res_12(64)
        self.conv8   = atr_res_12(64)
        self.maxpool4 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv9   = atr_res_12(64)
        self.conv10  = atr_res_12(64)
        self.maxpool5 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_e1 = nn.Conv3d(64, 1, 1) #fc layer
        
        
    def forward(self, x_init):
        #print('init_size',x.size())
        #x = F.elu(self.conv_i1(x_init))
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.maxpool1(x)
        
        x = F.elu(self.conv_i2(x_init))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool5(x)

        result = self.conv_e1(x) 

        return result
class Net_8q(nn.Module):
    def __init__(self):
        super(Net_8q, self).__init__()
        #done by quanv layer
        #self.conv_i1  = nn.Conv3d(3,   8, 1,padding=0)
        #self.conv1   = atr_res_12(8)
        #self.conv2   = atr_res_12(8)
        #self.maxpool1 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_i2  = nn.Conv3d(8,   64, 1,padding=0)
        self.conv3   = atr_res_12(64)
        self.conv4   = atr_res_12(64)
        self.maxpool2 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv5   = atr_res_12(64)
        self.conv6   = atr_res_12(64)
        self.maxpool3 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv7   = atr_res_12(64)
        self.conv8   = atr_res_12(64)
        self.maxpool4 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv9   = atr_res_12(64)
        self.conv10  = atr_res_12(64)
        self.maxpool5 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_e1 = nn.Conv3d(64, 1, 1) #fc layer
        
        
    def forward(self, x_init):
        #print('init_size',x.size())
        #x = F.elu(self.conv_i1(x_init))
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.maxpool1(x)
        
        x = F.elu(self.conv_i2(x_init))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool5(x)

        result = self.conv_e1(x) 

        return result
class Net_4q(nn.Module):
    def __init__(self):
        super(Net_4q, self).__init__()
        #done by quanv layer
        #self.conv_i1  = nn.Conv3d(3,   8, 1,padding=0)
        #self.conv1   = atr_res_12(8)
        #self.conv2   = atr_res_12(8)
        #self.maxpool1 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_i2  = nn.Conv3d(8,   64, 1,padding=0)
        self.conv3   = atr_res_12(64)
        self.conv4   = atr_res_12(64)
        self.maxpool2 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv5   = atr_res_12(64)
        self.conv6   = atr_res_12(64)
        self.maxpool3 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv7   = atr_res_12(64)
        self.conv8   = atr_res_12(64)
        self.maxpool4 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv9   = atr_res_12(64)
        self.conv10  = atr_res_12(64)
        self.maxpool5 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_e1 = nn.Conv3d(64, 1, 1) #fc layer
        
        
    def forward(self, x_init):
        #print('init_size',x.size())
        #x = F.elu(self.conv_i1(x_init))
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.maxpool1(x)
        
        x = F.elu(self.conv_i2(x_init))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool5(x)

        result = self.conv_e1(x) 

        return result

class atr_res_12(nn.Module):
    '''atrous_conv, dilation 1, dilation 2 w/resnet'''
    def __init__(self, ch):
        super(atr_res_12, self).__init__()
        self.conv11 = nn.Conv3d(ch,   ch, 3,padding=1,dilation=1)
        self.conv12 = nn.Conv3d(ch,   ch, 3,padding=2,dilation=2)
        self.conv13 = nn.Conv3d(2*ch, ch, 1,padding=0)
        self.bn1 = nn.BatchNorm3d(ch)

    def forward(self, x):
        x0 = x
        y1 = F.elu(self.conv11(x))
        y2 = F.elu(self.conv12(x))
        x  = F.elu(x0 + self.bn1(self.conv13(torch.cat((y1,y2),1))))
        
        return x

