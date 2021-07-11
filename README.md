# EfficientNet - Pytorch

Original Paper: https://arxiv.org/pdf/1905.11946.pdf

**Work in Progress**

This is my personal implementation of EfficientNet with Pytorch. I tried to build it only by reading the relevant Papers which will be referenced in the code. I start by implementing the baseline B0 network and go from there. Maybe I also try to build EfficientNetV2 in the long term. 

# TODOs

* DataLoader for ImageNet (the smaller Kaggle subset), CIFAR10/100 
* training script 
* utils for models (make all the code a bit more generic and clean)
* Data Augmentation (albumentations?) 
* evaluation script 
* inference demo notebook
* some explanation, how does this thing work and what makes it so efficient

# Basic Building Blocks

Besides the scaling strategy based on the compound scaling coefficient and neural architecture search, EfficientNet consists of a few basic building blocks which originate from older papers like MobileNet.

## Mobile Inverted Bottleneck

Original Paper: https://arxiv.org/pdf/1801.04381.pdf 

## Squeeze and Excitation Blocks 

Original Paper: https://arxiv.org/pdf/1709.01507.pdf

## Architecture 


