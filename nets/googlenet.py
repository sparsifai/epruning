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
    network_size = {'Conv2d-1':[64],'Conv2d-2':[192],'Conv2d-3':[384],'Conv2d-4':[256],'Conv2d-5':[256]} # for each layer: [number of featuremaps, size of inupt, size of output]
    return network_size

    
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)



class NetOrg(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(NetOrg, self).__init__()
        self.pre_layers = nn.Sequential(
                    nn.Conv2d(s_input_channel, 192, kernel_size=3, padding=1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(True),
                )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(2458624, n_classes) #1024

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)        
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)

        return output

class Net(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(s_input_channel, 6, kernel_size=5) # (1, 32, kernel=3, 1)   # Tensor size: #channels,#FMaps,#ConvImageSize(e.g. 28-3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # (9216, 128)
        self.fc2 = nn.Linear(120, 84) # (128, 10)
        self.fc3 = nn.Linear(84, n_classes) # (128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = torch.flatten(x, 1)        
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        logits = torch.relu(x)

        # dropout is good for limited data - test on medical images
        # Network capacity adjustment with dropout
        
        return output, sig1_x, sig2_x,logits



# class NetOrg(nn.Module):
#     def __init__(self, s_input_channel,num_classes, aux_logits=True, transform_input=False, init_weights=True,
#                  blocks=None):
#         super(NetOrg, self).__init__()
#         if blocks is None:
#             blocks = [BasicConv2d, Inception, InceptionAux]
#         assert len(blocks) == 3
#         conv_block = blocks[0]
#         inception_block = blocks[1]
#         inception_aux_block = blocks[2]

#         self.aux_logits = aux_logits
#         self.transform_input = transform_input

#         self.conv1 = conv_block(s_input_channel, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
#         self.conv2 = conv_block(64, 64, kernel_size=1)
#         self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

#         self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

#         self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

#         self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

#         if aux_logits:
#             self.aux1 = inception_aux_block(512, num_classes)
#             self.aux2 = inception_aux_block(528, num_classes)
#         else:
#             self.aux1 = None
#             self.aux2 = None

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(1024, num_classes)

#         if init_weights:
#             self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 import scipy.stats as stats
#                 X = stats.truncnorm(-2, 2, scale=0.01)
#                 values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
#                 values = values.view(m.weight.size())
#                 with torch.no_grad():
#                     m.weight.copy_(values)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _transform_input(self, x):
#         # type: (Tensor) -> Tensor
#         if self.transform_input:
#             x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
#             x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
#             x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
#             x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
#         return x

#     def _forward(self, x):
#         # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
#         # N x 3 x 224 x 224
#         x = self.conv1(x)
#         # N x 64 x 112 x 112
#         x = self.maxpool1(x)
#         # N x 64 x 56 x 56
#         x = self.conv2(x)
#         # N x 64 x 56 x 56
#         x = self.conv3(x)
#         # N x 192 x 56 x 56
#         x = self.maxpool2(x)

#         # N x 192 x 28 x 28
#         x = self.inception3a(x)
#         # N x 256 x 28 x 28
#         x = self.inception3b(x)
#         # N x 480 x 28 x 28
#         x = self.maxpool3(x)
#         # N x 480 x 14 x 14
#         x = self.inception4a(x)
#         # N x 512 x 14 x 14
#         # aux1 = torch.jit.annotate(Optional[Tensor], None)
#         # if self.aux1 is not None:
#         #     if self.training:
#         #         aux1 = self.aux1(x)

#         x = self.inception4b(x)
#         # N x 512 x 14 x 14
#         x = self.inception4c(x)
#         # N x 512 x 14 x 14
#         x = self.inception4d(x)
#         # N x 528 x 14 x 14
#         # aux2 = torch.jit.annotate(Optional[Tensor], None)
#         # if self.aux2 is not None:
#         #     if self.training:
#         #         aux2 = self.aux2(x)

#         x = self.inception4e(x)
#         # N x 832 x 14 x 14
#         x = self.maxpool4(x)
#         # N x 832 x 7 x 7
#         x = self.inception5a(x)
#         # N x 832 x 7 x 7
#         x = self.inception5b(x)
#         # N x 1024 x 7 x 7

#         x = self.avgpool(x)
#         # N x 1024 x 1 x 1
#         x = torch.flatten(x, 1)
#         # N x 1024
#         x = self.dropout(x)
#         x = self.fc(x)
#         # N x 1000 (num_classes)
#         return x, aux2, aux1

#     @torch.jit.unused
#     def eager_outputs(self, x, aux2, aux1):
#         # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
#         if self.training and self.aux_logits:
#             return _GoogLeNetOutputs(x, aux2, aux1)
#         else:
#             return x

#     def forward(self, x):
#         # type: (Tensor) -> GoogLeNetOutputs
#         x = self._transform_input(x)
#         x, aux1, aux2 = self._forward(x)
#         aux_defined = self.training and self.aux_logits
#         if torch.jit.is_scripting():
#             if not aux_defined:
#                 warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
#             return GoogLeNetOutputs(x, aux2, aux1)
#         else:
#             return self.eager_outputs(x, aux2, aux1)


# class Inception(nn.Module):

#     def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
#                  conv_block=None):
#         super(Inception, self).__init__()
#         if conv_block is None:
#             conv_block = BasicConv2d
#         self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

#         self.branch2 = nn.Sequential(
#             conv_block(in_channels, ch3x3red, kernel_size=1),
#             conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
#         )

#         self.branch3 = nn.Sequential(
#             conv_block(in_channels, ch5x5red, kernel_size=1),
#             # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
#             # Please see https://github.com/pytorch/vision/issues/906 for details.
#             conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
#         )

#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
#             conv_block(in_channels, pool_proj, kernel_size=1)
#         )

#     def _forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)

#         outputs = [branch1, branch2, branch3, branch4]
#         return outputs

#     def forward(self, x):
#         outputs = self._forward(x)
#         return torch.cat(outputs, 1)


# class InceptionAux(nn.Module):

#     def __init__(self, in_channels, num_classes, conv_block=None):
#         super(InceptionAux, self).__init__()
#         if conv_block is None:
#             conv_block = BasicConv2d
#         self.conv = conv_block(in_channels, 128, kernel_size=1)

#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
#         x = F.adaptive_avg_pool2d(x, (4, 4))
#         # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
#         x = self.conv(x)
#         # N x 128 x 4 x 4
#         x = torch.flatten(x, 1)
#         # N x 2048
#         x = F.relu(self.fc1(x), inplace=True)
#         # N x 1024
#         x = F.dropout(x, 0.7, training=self.training)
#         # N x 1024
#         x = self.fc2(x)
#         # N x 1000 (num_classes)

#         return x


# class BasicConv2d(nn.Module):

#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return F.relu(x, inplace=True)