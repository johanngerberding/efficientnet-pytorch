import os
from albumentations.augmentations.geometric.resize import Resize
from scipy.sparse.construct import rand
import torch 
import numpy as np
import torch.nn as nn 

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from efficientnet.model import EfficientNet
from efficientnet.utils import get_n_params, get_config
from dataset.cifar import CIFAR10

import albumentations as A 
from albumentations.pytorch import ToTensorV2


def train(train_loader, model, criterion, optimizer, epoch, device):
    return NotImplementedError


def validate(train_loader, model, criterion, epoch, device):
    return NotImplementedError


def main():
    # training parameters
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    bn_momentum = 0.99
    decay = 0.9
    weight_decay = 0.00001
    num_workers = 4 
    epochs = 10

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: {}".format(device))

    config = get_config('efficientnet-b0')
    net = EfficientNet(config=config, num_classes=10)
    net.to(device)

    test_tensor = torch.randn(1,3,64,64)
    test_tensor = test_tensor.to(device)
    out = net(test_tensor)
    print(out.size())
    
    #print(net)
    num_params = get_n_params(net)
    print(num_params)

    cifar10_root = "/home/johann/dev/efficientnet-pytorch/data/cifar10"
    meta_path = '/media/data/CIFAR/cifar-10-batches-py/batches.meta'
    train_imgs_pkl = os.path.join(cifar10_root, 'train/train_imgs.pkl')
    train_labels_pkl = os.path.join(cifar10_root, 'train/train_labels.pkl')
    
    train_transform = A.Compose(
        [
            A.Resize((64,64)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

    )
    test_transform = A.Compose(
        [
            A.Resize((64,64)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = CIFAR10(meta_path, cifar10_root, train_imgs_pkl, train_labels_pkl, True, train_transform)
    
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


    for epoch in range(1, (epochs+1)):
        train()
        validate()



if __name__ == '__main__':
    main()
