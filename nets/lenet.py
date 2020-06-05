from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time,os,sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import hsummary
import torchvision
from numpy import unique
from scipy.stats import entropy as scipy_entropy

def networksize(n_classes):
    network_size = {'Conv2d': [[6],[16]], 'fc':[[400],[120],[84],[n_classes]]} # for each layer: [number of featuremaps, size of inupt, size of output]
    kernels = [5,5] 
    return network_size, kernels


class NetOrg(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(NetOrg, self).__init__()
        self.conv1 = nn.Conv2d(s_input_channel, 6, kernel_size=5) # (1, 32, kernel=3, 1)   # Tensor size: #channels,#FMaps,#ConvImageSize(e.g. 28-3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(400, 120) # 32:400 28:256   (9216, 128)16 * 5 * 5
        self.fc2 = nn.Linear(120, 84) # (128, 10)
        self.fc3 = nn.Linear(84, n_classes) # (128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        # net_signals['Conv2d-1'] = x
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = torch.relu(x)
        # net_signals['Conv2d-2'] = x
        x = F.max_pool2d(x, kernel_size=2)

        x = torch.flatten(x, 1)        
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        # net_signals['fc-1'] = x

        # x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        # net_signals['fc-2'] = x
        # x2 = x
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output,x


class Net(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(s_input_channel, 6, kernel_size=5) # (1, 32, kernel=3, 1)   # Tensor size: #channels,#FMaps,#ConvImageSize(e.g. 28-3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(400, 120) # (9216, 128) 16 * 5 * 5
        self.fc2 = nn.Linear(120, 84) # (128, 10)
        self.fc3 = nn.Linear(84, n_classes) # (128, 10)

    def forward(self, x, states_reshaped_):
        self.states_reshaped = states_reshaped_
        net_signals = {} # dictionary of activation signals
        x = self.conv1(x)
        x = torch.relu(x)
        x1 = x
        net_signals['Conv2d-1'] = x1

        x = self.states_reshaped['Conv2d-1'] * x
        x = F.max_pool2d(x, kernel_size=2)
        x1 = F.max_pool2d(x1, kernel_size=2)


        x = self.conv2(x)
        x = torch.relu(x)
        x = self.states_reshaped['Conv2d-2'] * x
        x = F.max_pool2d(x, kernel_size=2)

        x1 = self.conv2(x1)
        x1 = torch.relu(x1)
        net_signals['Conv2d-2'] = x1
        x1 = F.max_pool2d(x1, kernel_size=2)



        x = torch.flatten(x, 1)
        x = self.states_reshaped['fc-1'] * x

        x1 = torch.flatten(x1, 1)
        net_signals['fc-0'] = x1        
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x1 = self.fc1(x1)
        x1 = torch.relu(x1)
        net_signals['fc-1'] = x1

        x = self.states_reshaped['fc-2'] * x
        x = self.fc2(x)
        x = torch.relu(x)

        x1 = self.fc2(x1)
        x1 = torch.relu(x1)
        net_signals['fc-2'] = x1


        x = self.states_reshaped['fc-3'] * x
        x = self.fc3(x)

        x1 = self.fc3(x1)
        net_signals['fc-3'] = x1

        output = F.log_softmax(x, dim=1)
        logits = x

        # dropout is good for limited data - test on medical images
        # Network capacity adjustment with dropout
        
        return output, logits, net_signals #,d1,d2


def gradient_mask(model,best_stateI,device):

    for indx, p in enumerate(model.parameters()):
        c = 0
        if indx%2==0: # just layers - bias included
            shp = p.grad.shape
            if len(p.grad.shape)==4: # it is convolution
                s = p.grad.shape[0]
                msk = np.zeros((shp))
                states_ = best_stateI[c:c+s]
                for ind in range(s):
                    msk[ind,:,:,:] = states_[ind]
                p.grad = p.grad*torch.from_numpy(msk).float().to(device)
                for indx_bias, p_bias in enumerate(model.parameters()):
                    if indx_bias==indx+1:
                        shp = p_bias.grad.shape
                        msk = np.zeros((shp))
                        states_ = best_stateI[c:c+s]
                        for ind in range(s):
                            msk[ind] = states_[ind]
                        p_bias.grad = p_bias.grad*torch.from_numpy(msk).float().to(device)
                        break
            elif len(p.grad.shape)==2: # it is dense 
                s = p.grad.shape[1]
                msk = np.zeros((shp))
                states_ = best_stateI[c:c+s]
                for ind in range(s):
                    msk[:,ind] = states_[ind]
                p.grad = p.grad*torch.from_numpy(msk).float().to(device)
                flat_dense_delta = 400
                for indx_bias, p_bias in enumerate(model.parameters()):
                    if indx_bias==indx+1:
                        s = p_bias.grad.shape[0]
                        shp = p_bias.grad.shape
                        msk = np.zeros((shp))
                        states_ = best_stateI[flat_dense_delta+c:flat_dense_delta+c+s]
                        for ind in range(s):
                            msk[ind,] = states_[ind]
                        p_bias.grad = p_bias.grad*torch.from_numpy(msk).float().to(device)
                        break
        c+=s
    return None