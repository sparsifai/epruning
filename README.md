# EDropout
Energy-based Dropout and Pruning of Deep Neural Networks

This page contains codes and description on the EDropout method.

## Requirements


## Structure

## Datasets
The following setup of benchmark datasets are used: 

(i) Fashion (gray images in 10 classes, 54k train, 6k validation, and 10k test);

(ii) Kuzushiji (gray images in 10 classes, 54k train, 6k validation, and 10k test); 

(iii) CIFAR-10 (color images in 10 classes, 45k train, 5k validation, and 10k test);

(iv) CIFAR-100 (color images in 100 classes, 45k train, 5k validation, and 10k test);

(v) Flowers (102 flower categories; each class has between 40 and 258 images; 10 images from each class for validation and 10 for test). 

The horizontal flip and Cutout augmentation methods are used for training on CIFAR and Flowers datasets. Input images are resized to 32x32 for ResNets and for 224x224 AlexNet and SqueezeNetv1.1. 

## Models
### ResNets

### SqueezeNet

### AlexNet

### Deep Compression

### EDropout


## Hyperparameters


## Training


We have conducted a high level hyper-parameters tuning and found the following ones:

- Learning rate: Initial leanring rate of 1 with adaptive step learning rate decaly with gamma 0.1 at every 50 epoch 
- Optimizer: Adadelta with rho=0.9, eps=1e-06, weight_decay=0.00001
- Batch-size: 128
- Validation dataset: 10% of the training dataset selected randomly
- Number of candidate states: 8
- Early convergence threshold: 100
- Number of epochs: 200
- Initial probabilty of binary states: 0.5
- Augmentation: CropOut for CIFAR and Flowers datasets

Some hyper-parameters analysis are provided in the paper.

### How to Train
`python3 edropout.py`

Parameters inside edropout.py:   
dataset = {'fashion','kuzushiji','cifar10','cifar100','flowers'}    
nnmodel = {'resnet18','resnet34','resnet50','resnet101'}   
model = {'ising','simple'} # ising: edropout method; simple: original model   

## Results


## Docker


## Parallel Implementation
The current version of the optimization phase is written in NumPy as a POC. A parallel version will be implemented and added in PyTorch very soon.  

## Contact
Please send your feedback and comments to sparsifai.ai@gmail.com

## Citation
The paper is submitted to Neurips 2020. Please check later.


## References


