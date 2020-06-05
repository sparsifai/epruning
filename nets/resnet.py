'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable



# def networksize(resnet_type,n_classes):
#     if resnet_type == 'resnet20':
#         network_size = {'Conv2d': [[6],[16]], 'fc':[[400],[120],[84],[n_classes]]} # for each layer: [number of featuremaps, size of inupt, size of output]
#         kernels = [5,5] 
#     return network_size, kernels




__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, net_signals, id, counter, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.net_signals = net_signals
        self.shortcut = nn.Sequential()
        self.stride = stride
        self.id = id
        self.counter = counter
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # print('stride',self.stride)

        out = F.relu(self.bn1(self.conv1(x)))
        key = 'layer'+str(self.id)+'.'+str(self.counter)+'.conv1' #,out.shape)
        self.net_signals[key] = out

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        key = 'layer'+str(self.id)+'.'+str(self.counter)+'.conv2' #,out.shape)

        self.net_signals[key] = out

        return out


class ResNetOrg(nn.Module):
    def __init__(self, s_input_channel, n_classes, resnet_type):
        super(ResNetOrg, self).__init__()
        block = BasicBlock
        num_classes = n_classes
        if resnet_type == 'resnet20':
            num_blocks = [3, 3, 3]
        elif resnet_type == 'resnet56':
            num_blocks = [9, 9, 9]

        self.in_planes = 16
        self.net_signals = {}
        self.conv1 = nn.Conv2d(s_input_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            id = None
            counter= None
            layers.append(block(self.in_planes, self.net_signals, id, counter, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        logits = out
        # out = F.log_softmax(out, dim=1)

        return out, logits


class ResNet(nn.Module):
    def __init__(self, s_input_channel, n_classes, resnet_type):
        super(ResNet, self).__init__()
        block = BasicBlock
        num_classes = n_classes
        if resnet_type == 'resnet20':
            num_blocks = [3, 3, 3]
        elif resnet_type == 'resnet56':
            num_blocks = [9, 9, 9]

        self.net_signals = {}

        self.in_planes = 16

        self.conv1 = nn.Conv2d(s_input_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, id=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, id=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, id=3)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, id):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        counter = 0
        for stride in strides:

            layers.append(block(self.in_planes, self.net_signals, id, counter, planes, stride))
            self.in_planes = planes * block.expansion
            counter+=1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.net_signals['conv1'] = out
        # print('conv1')
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.net_signals['linear'] = out

        out = self.linear(out)
        # print('linear',out.shape)
        # print('dddd',self.net_signals['linear'].shape)
        # print(self.net_signals.keys())
        net_energy = out # logits value
        # out = F.log_softmax(out, dim=1)

        return out, net_energy, self.net_signals



def gradient_mask(model,best_stateI,device):

    # ss = model._modules['conv1'].grad
    # print(len(model.parameters()))
    # for indx, p in enumerate(model.parameters()):
    #     print(indx,p.shape)
    # 0: 'conv1'
    convs_indx = [i for i in range(3,57,3)]
    # print(convs_indx)
    # 57: 'linear'
    c = 0

    for indx, p in enumerate(model.parameters()):
        # print(indx,p.shape)

        shp = p.grad.shape
        if indx == 0:
            s = p.grad.shape[0]
            msk = np.zeros((shp))
            states_ = best_stateI[c:c+s]
            for ind in range(s):
                msk[ind,:,:,:] = states_[ind]
            p.grad = p.grad*torch.from_numpy(msk).float().to(device)
        elif indx == 57:
            s = p.grad.shape[1]
            msk = np.zeros((shp))
            states_ = best_stateI[c:c+s]
            for ind in range(s):
                msk[:,ind] = states_[ind]
            p.grad = p.grad*torch.from_numpy(msk).float().to(device)
        elif indx in convs_indx:
            s = p.grad.shape[0]
            msk = np.zeros((shp))
            states_ = best_stateI[c:c+s]
            for ind in range(s):
                msk[ind,:,:,:] = states_[ind]
            p.grad = p.grad*torch.from_numpy(msk).float().to(device)
        c+=s
    return None







class BasicBlockPrune(nn.Module):
    expansion = 1

    def __init__(self, in_planes, net_signals, id, cfg, counter, planes, stride=1, option='A'):
        super(BasicBlockPrune, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, cfg, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.conv2 = nn.Conv2d(cfg, cfg, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg)
        self.net_signals = net_signals
        self.shortcut = nn.Sequential()
        self.stride = stride
        self.id = id
        self.counter = counter
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # print('stride',self.stride)

        out = F.relu(self.bn1(self.conv1(x)))
        key = 'layer'+str(self.id)+'.'+str(self.counter)+'.conv1' #,out.shape)
        self.net_signals[key] = out

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        key = 'layer'+str(self.id)+'.'+str(self.counter)+'.conv2' #,out.shape)

        self.net_signals[key] = out

        return out



class ResNetPrune(nn.Module):
    def __init__(self, s_input_channel, n_classes, resnet_type,cfg):
        super(ResNetPrune, self).__init__()
        block = BasicBlockPrune
        num_classes = n_classes
        if resnet_type == 'resnet20':
            num_blocks = [3, 3, 3]
        elif resnet_type == 'resnet56':
            num_blocks = [9, 9, 9]

        self.net_signals = {}

        self.in_planes = 16
        self.conv1 = nn.Conv2d(s_input_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        count = 0
        # print(cfg,num_blocks[0])
        self.layer1 = self._make_layer(block, 16, num_blocks[0], cfg[:num_blocks[0]], stride=1, id=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], cfg[count:count+num_blocks[1]], stride=2, id=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], cfg[count:count+num_blocks[1]], stride=2, id=3)
        self.linear = nn.Linear(cfg[-1], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, cfg, stride, id):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        counter = 0
        for stride in strides:

            layers.append(block(self.in_planes, self.net_signals, id, cfg[stride], counter, planes, stride))
            self.in_planes = planes * block.expansion
            counter+=1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.net_signals['conv1'] = out
        # print('conv1')
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.net_signals['linear'] = out

        out = self.linear(out)
        # print('linear',out.shape)
        # print('dddd',self.net_signals['linear'].shape)
        # print(self.net_signals.keys())
        net_energy = out # logits value
        # out = F.log_softmax(out, dim=1)

        return out, net_energy



# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])


# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])


# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])


# def resnet56():
#     return ResNet(BasicBlock, [9, 9, 9])


# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])


# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])


# def test(net):
#     import numpy as np
#     total_params = 0

#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


# if __name__ == "__main__":
#     for net_name in __all__:
#         if net_name.startswith('resnet'):
#             print(net_name)
#             test(globals()[net_name]())
#             print()