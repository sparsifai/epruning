
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
# import matplotlib.pyplot as plt
import copy
# import tensorflow as tf
import torchvision
from numpy import unique
from scipy.stats import entropy as scipy_entropy

def traindata(kwargs,args,input_size,valid_percentage,dataset):
    if dataset == 'mnist':
        root = '/data/mnist/images/train'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                # transforms.RandomResizedCrop(input_size[1]),
                transforms.Grayscale(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.,), (1.,))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
                ])
    elif dataset == 'fashion':
        root = '/data/fashion/images/train'  
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                # transforms.RandomResizedCrop(input_size[1]),
                transforms.Grayscale(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.,), (1.,))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])    
    
    elif dataset == 'kuzushiji':
        root = '/data/kuzushiji/images/train'  
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.Grayscale(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.,), (1.,))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
    elif dataset == 'flowers':
        root = '/data/flowers/images/train'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(input_size[1], padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0,180), resample=False, expand=False, center=None, fill=0),
                transforms.ToTensor(),
                # transforms.Normalize((0.,), (1.,))
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    elif dataset == 'cifar10':
        root = '/data/cifar10/images/train'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.,), (1.,))
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
    elif dataset == 'cifar100':
        root = '/data/cifar100/images/train'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((0,180), resample=False, expand=False, center=None, fill=0),                transforms.ToTensor(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
    
    hymenoptera_dataset = datasets.ImageFolder(root=root,
                                            transform=data_transform)
    
    ## Training Validation Split
    valid_size = int(valid_percentage*len(hymenoptera_dataset)) # in number of batches
    train_size = len(hymenoptera_dataset) - valid_size
    subsetTrain, subsetValid = torch.utils.data.random_split(hymenoptera_dataset, [train_size, valid_size])
    ### Train data  
    train_loader = torch.utils.data.DataLoader(subsetTrain,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=4)
    ### Valid data
    valid_loader = torch.utils.data.DataLoader(subsetValid,
                                            batch_size=args.batch_size, shuffle=True,
                                            num_workers=4)

    return train_loader,valid_loader

def testdata(kwargs,args,input_size,dataset):
    if dataset == 'mnist':
        root = '/data/mnist/images/test'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.,), (1.,))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
    elif dataset == 'fashion':
        root = '/data/fashion/images/test'        
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.,), (1.,))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])   

    elif dataset == 'kuzushiji':
        root = '/data/kuzushiji/images/test'  
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.Grayscale(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.,), (1.,))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])

    elif dataset == 'flowers':
        root = '/data/flowers/images/test'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    elif dataset == 'cifar10':
        root = '/data/cifar10/images/test'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])), 
                transforms.ToTensor(),
                # transforms.Normalize((0.,), (1.,))
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
    elif dataset == 'cifar100':
        root = '/data/cifar100/images/test'
        data_transform = transforms.Compose([
                transforms.Resize((input_size[1],input_size[2])),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((0,180), resample=False, expand=False, center=None, fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])

    hymenoptera_dataset = datasets.ImageFolder(root=root,
                                            transform=data_transform)
    test_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=4)

    return test_loader

def dataset_specs(dataset):
    if dataset == 'mnist':
        n_classes = 10
        s_input_channel = 1
    elif dataset == 'fashion':
        n_classes = 10
        s_input_channel = 1
    elif dataset == 'kuzushiji':
        n_classes = 10
        s_input_channel = 1
    elif dataset == 'flowers':
        n_classes = 102
        s_input_channel = 3
    elif dataset == 'cifar10':
        n_classes = 10
        s_input_channel = 3
    elif dataset == 'cifar100':
        n_classes = 100
        s_input_channel = 3
    return n_classes, s_input_channel