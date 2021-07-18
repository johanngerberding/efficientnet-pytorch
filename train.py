import os
import torch 
import yaml 
import numpy as np
import torch.nn as nn 
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime 

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from efficientnet.model import EfficientNet
from efficientnet.utils import get_n_params, get_config
from dataset.cifar import CIFAR10

import albumentations as A 
from albumentations.pytorch import ToTensorV2


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def calc_acc(output, labels):
    "Calculate accuracy of predictions"
    pred_softmax = torch.log_softmax(output, dim=1)
    _, preds = torch.max(pred_softmax, dim=1)
    correct = (preds == labels).float()
    acc = correct.sum() / len(correct)
    acc = torch.round(acc * 100)
    
    return acc


def train(train_loader, model, criterion, optimizer, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, labels) in enumerate(stream, start=1):
        images = images.to(device)
        labels = labels.to(device)        
        output = model(images)
        loss = criterion(output, labels)
        acc = calc_acc(output, labels)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {}. Train.      {}".format(epoch, metric_monitor)
        )



def validate(val_loader, model, criterion, epoch, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, labels) in enumerate(stream, start=1):
            images = images.to(device)
            labels = labels.to(device)        
            output = model(images)
            loss = criterion(output, labels)
            acc = calc_acc(output, labels)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", acc)
            stream.set_description(
                "Epoch: {}. Validation. {}".format(epoch, metric_monitor)
            )



def main():
    # training parameters
    """
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    bn_momentum = 0.99
    decay = 0.9
    weight_decay = 0.00001
    num_workers = 4 
    epochs = 25
    resume_from = None
    work_dir = ''
    save_interval = 0
    model_name = 'efficientnet-b0'
    """
    with open("config.yaml", 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print(cfg)

    now = datetime.now()
    now_str = now.strftime("%d_%m_%y")
    
    if cfg.work_dir == '':
        work_dir = "experiments/{}".format(now_str)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: {}".format(device))

    config = get_config(cfg.model_name)
    net = EfficientNet(config=config, num_classes=10)
    net.to(device)

    #test_tensor = torch.randn(1,3,64,64)
    #test_tensor = test_tensor.to(device)
    #out = net(test_tensor)
    #num_params = get_n_params(net)

    cifar10_root = "/home/johann/dev/efficientnet-pytorch/data/cifar10"
    meta_path = '/home/johann/dev/efficientnet-pytorch/data/cifar10/cifar-10-batches-py/batches.meta'
    train_imgs_pkl = os.path.join(cifar10_root, 'train/train_imgs.pkl')
    train_labels_pkl = os.path.join(cifar10_root, 'train/train_labels.pkl')
    
    train_transform = A.Compose(
        [
            A.Resize(64,64),
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
            A.Resize(64,64),
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
                                  batch_size=cfg.train_batch_size, 
                                  shuffle=True, 
                                  num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.val_batch_size, 
                                shuffle=True, 
                                num_workers=cfg.num_workers)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)


    for epoch in range(1, (cfg.epochs+1)):

        train(train_dataloader, net, criterion, optimizer, epoch, device)
        validate(val_dataloader, net, criterion, epoch, device)
        
        if cfg.save_interval != 0 and epoch % cfg.save_interval == 0:
            torch.save(net.state_dict(), os.path.join(work_dir, 'epoch_{}.pth'.format(epoch)))

    torch.save(net.state_dict(), os.path.join(work_dir, 'final_{}.pth'.format(epoch)))


if __name__ == '__main__':
    main()
