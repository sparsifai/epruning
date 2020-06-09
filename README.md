# EDropout
Energy-based Dropout and Pruning of Deep Neural Networks

This page contains codes and description on the EDropout method.

## Requirements
- Python 3.7+
- PyTorch 1.3+
- torchvision
- CUDA 10+
- numpy

## Datasets
The following setup of benchmark datasets are used: 

(i) Fashion (gray images in 10 classes, 54k train, 6k validation, and 10k test);

(ii) Kuzushiji (gray images in 10 classes, 54k train, 6k validation, and 10k test); 

(iii) CIFAR-10 (color images in 10 classes, 45k train, 5k validation, and 10k test);

(iv) CIFAR-100 (color images in 100 classes, 45k train, 5k validation, and 10k test);

(v) Flowers (102 flower categories; each class has between 40 and 258 images; 10 images from each class for validation and 10 for test). 

The horizontal flip and Cutout augmentation methods are used for training on CIFAR and Flowers datasets. Input images are resized to 32x32 for ResNets and for 224x224 AlexNet and SqueezeNetv1.1. 

A sample of data structur is presented in data directory for Fashion dataset.

The dataloader file is uner untils directory. The Fashion and Kuzushiji are normalized in [0,1] and the other images are normalized in this setup: (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Models
The models are located in the nets directory. We mainly used the standard torchvision source codes: 

### ResNets

https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

### SqueezeNet

https://pytorch.org/hub/pytorch_vision_squeezenet/

### AlexNet

https://pytorch.org/hub/pytorch_vision_alexnet/

### Deep Compression

https://github.com/mightydeveloper/Deep-Compression-PyTorch

## Hyperparameters

We have conducted a high level hyper-parameters search in following space:

- Initial learning rate: {1,0.1,0.01}
- Adaptive learning rate gamma: {0.1,0.5,0.9}
- Learnign rate step: {25,50}
- Batch size: {64,128}
- Optimizer: {SGD, Aadadelta}
- Weight decay: {0.00001,0.000001}
- Number of epochs: {200,400}
- Early convergence threshold: {50,100,150,200}
- Initial probabilty of binary states: {0.2,0.4,0.6,0.8}


The parameters for most edropout experiemnts are:

- Learning rate: Initial leanring rate of 1 with adaptive step learning rate decaly with gamma 0.1 at every 50 epoch 
- Optimizer: Adadelta with rho=0.9, eps=1e-06, weight_decay=0.00001
- Batch-size: 128
- Validation dataset: 10% of the training dataset selected randomly
- Number of candidate states: 8
- Early convergence threshold: 100
- Number of epochs: 200
- Initial probabilty of binary states: 0.5
- Augmentation: CropOut + RandomRotation in [0,180] for CIFAR and Flowers datasets

Some hyper-parameters analysis are provided in the paper.

### How to Train
`python3 edropout.py`

Parameters inside edropout.py:   
dataset = {'fashion','kuzushiji','cifar10','cifar100','flowers'}    
nnmodel = {'resnet18','resnet34','resnet50','resnet101'}   
model = {'ising','simple'} # ising: edropout method; simple: original model   

## Results
The results are average of five independant executions. More results are provided in the paper.
![alt text](https://github.com/sparsifai/edropout/blob/master/png/k.png)
![alt text](https://github.com/sparsifai/edropout/blob/master/png/f.png)


## Docker
A docker container will be pushed asap.

## Parallel Implementation
The current version of the optimization phase is written in NumPy as a POC for fast implementation. A parallel version will be implemented and added in PyTorch very soon. The executing time using numpy on a single RTX GPU on the Flowers dataset with 8 candidate state vectors is as follows:

| Model        | EDropout/Original  | Number of States
| ------------- |:-------------:| -----:|
| resnet18     | 17.21x | 6208 |
| resnet34     | 25.12x      | 9920  |
| resnet50  | 19.63x    |  32448  |
resnet101 | 22.56x | 58560
AlexNet | 28.97x | 18662
SqueezeNet | 1.33x| 3558


Again, we need to emphasize that this is a quick implementation and a parallel version will be uploaded asap. 


## Contact
Please send your feedback and comments to sparsifai.ai@gmail.com

## Citation
The paper is available at: https://arxiv.org/abs/2006.04270

@misc{salehinejad2020edropout,     
    title={EDropout: Energy-Based Dropout and Pruning of Deep Neural Networks},     
    author={Hojjat Salehinejad and Shahrokh Valaee},     
    year={2020},     
    eprint={2006.04270},      
    archivePrefix={arXiv},     
    primaryClass={cs.LG}     
    
}


## References


