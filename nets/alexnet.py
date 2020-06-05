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

def networksize():
    network_size = {'Conv2d-1':[64],'Conv2d-2':[192],'Conv2d-3':[384],'Conv2d-4':[256],'Conv2d-5':[256],'fc-1':[4096],'fc-2':[4096]} # for each layer: [number of featuremaps, size of inupt, size of output]
    return network_size

class NetOrg(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(NetOrg, self).__init__()
        self.conv1 = nn.Conv2d(s_input_channel, 64, kernel_size=11, stride=4, padding=2) # (1, 32, kernel=3, 1)   # Tensor size: #channels,#FMaps,#ConvImageSize(e.g. 28-3)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2) 
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1) 
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.avgpooling = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096) # (9216, 128)
        self.fc2 = nn.Linear(4096, 4096) # (128, 10)
        self.fc3 = nn.Linear(4096, n_classes) # (128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv3(x)
        x = torch.relu(x)

        x = self.conv4(x)
        x = torch.relu(x)

        x = self.conv5(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.avgpooling(x)

        x = torch.flatten(x, 1)        
        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)

        return output 


class Net(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(s_input_channel, 64, kernel_size=11, stride=4, padding=2) # (1, 32, kernel=3, 1)   # Tensor size: #channels,#FMaps,#ConvImageSize(e.g. 28-3)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2) 
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1) 
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.avgpooling = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096) # (9216, 128)
        self.fc2 = nn.Linear(4096, 4096) # (128, 10)
        self.fc3 = nn.Linear(4096, n_classes) # (128, 10)

    def forward(self, x, states_reshaped_):
        self.states_reshaped = states_reshaped_
        net_signals = {} # dictionary of activation signals
        # self.states_conv1_x = self.states_reshaped['conv2d_1']
        # self.states_conv2_x = self.states_reshaped['conv2_x']

        x = self.conv1(x)
        x = torch.relu(x)
        net_signals['Conv2d-1'] = x
        x = self.states_reshaped['Conv2d-1'] * x
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        net_signals['Conv2d-2'] = x
        x = self.states_reshaped['Conv2d-2'] * x
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv3(x)
        x = torch.relu(x)
        net_signals['Conv2d-3'] = x
        x = self.states_reshaped['Conv2d-3'] * x

        x = self.conv4(x)
        x = torch.relu(x)
        net_signals['Conv2d-4'] = x
        x = self.states_reshaped['Conv2d-4'] * x

        x = self.conv5(x)
        x = torch.relu(x)
        net_signals['Conv2d-5'] = x
        x = self.states_reshaped['Conv2d-5'] * x
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.avgpooling(x)

        x = torch.flatten(x, 1)      
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        net_signals['fc-1'] = x
        x = self.states_reshaped['fc-1'] * x

        # x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        net_signals['fc-2'] = x        
        x = self.states_reshaped['fc-2'] * x

        x = self.fc3(x)
        logits = torch.relu(x)

        output = F.log_softmax(x, dim=1)
        # dropout is good for limited data - test on medical images
        # Network capacity adjustment with dropout
        
        return output, logits, net_signals #sig1_x, sig2_x,
