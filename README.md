# EDropout
Energy-based Dropout and Pruning of Deep Neural Networks

This page contains codes and description on the EDropout method.

## Requirements


## Structure

## Datasets
The following setup of benchmark datasets are used: 
(i) Fashion-MNIST (gray images in 10 classes, 54k train, 6k validation, and 10k test)~\citep{xiao2017}
(ii) Kuzushiji-MNIST (gray images in 10 classes, 54k train, 6k validation, and 10k test)~\citep{clanuwat2018deep}; (iii) CIFAR-10 (color images in 10 classes, 45k train, 5k validation, and 10k test)~\citep{krizhevsky2009learning}, (iv)
CIFAR-100 (color images in 100 classes, 45k train, 5k validation, and 10k test)~\citep{krizhevsky2009learning}, and (v) Flowers (102 flower categories; each class has between 40 and 258 images; 10 images from each class for validation and 10 for test)~\citep{nilsback2008automated}. The horizontal flip and Cutout~\citep{devries2017improved} augmentation methods are used for training on CIFAR and Flowers datasets. Input images are resized to $32\times32$ for ResNets and for $224\times224$ AlexNet~\citep{krizhevsky2012imagenet} and SqueezeNet v1.1~\citep{iandola2016squeezenet}. 

## Models
### ResNets

### SqueezeNet

### AlexNet

### Deep Compression

## Training


## Results


## Parallel Implementation
The current version of the optimization phase is written in NumPy as a POC. A parallel version will be implemented and added in PyTorch very soon.  

## Citation
The paper is submitted to Neurips 2020. Please check later.
