import os
import torch 
import yaml 
import argparse
import math 
import numpy as np
import torch.nn as nn 
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime 

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from efficientnet.model import EfficientNet
from efficientnet.utils import ModelParams
from dataset.imagenet import ImageNet

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


def train(train_loader, model, criterion, optimizer, epoch, device, scaler):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, labels) in enumerate(stream, start=1):
        images = images.to(device)
        labels = labels.to(device)  

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():     
            output = model(images)
            loss = criterion(output, labels)

        acc = calc_acc(output, labels)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", acc)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
            with torch.cuda.amp.autocast(enabled=True):  
                output = model(images)
                loss = criterion(output, labels)

            acc = calc_acc(output, labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", acc)
            stream.set_description(
                "Epoch: {}. Validation. {}".format(epoch, metric_monitor)
            )


def load_config(path):
    with open(path, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


class Trainer:
    NotImplementedError


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0,
                 warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + (increase/10)

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                    math.pi *
                    (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, help="name of the efficientnet model, e.g. 'efficientnet-b0'", required=True)
    parser.add_argument('--config', type=str, help="path to config yaml", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    now = datetime.now()
    now_str = now.strftime("%d_%m_%y")
    
    if cfg['work_dir'] == '':
        work_dir = "experiments/{}".format(now_str)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: {}".format(device))

    model_params = ModelParams(args.name, 
                               num_classes=cfg['num_classes'])

    net = EfficientNet(model_params)
    net.to(device)

    root = cfg['root']
    root_imgs = os.path.join(root, "ILSVRC", "Data", "CLS-LOC")
    train_imgs_root = os.path.join(root_imgs, 'train')
    label_file = os.path.join(root, 'LOC_synset_mapping.txt')

    train_transform = A.Compose(
        [
            A.Resize(model_params.img_size, model_params.img_size),
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
            A.Resize(model_params.img_size,model_params.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = ImageNet(train_imgs_root, "", label_file, True, train_transform)
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg['train_batch_size'], 
                                  shuffle=True, 
                                  num_workers=cfg['num_workers'])
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg['val_batch_size'], 
                                shuffle=True, 
                                num_workers=cfg['num_workers'])

    criterion = nn.CrossEntropyLoss().to(device)

    # here I don't know exactly what the authors mean by "optimizer with decay 0.9"
    # do they mean the smoothing constant alpha?
    optimizer = torch.optim.RMSprop(net.parameters(), 
                                    lr=cfg['learning_rate'], 
                                    momentum=cfg['momentum'], 
                                    weight_decay=cfg['weight_decay'])

    # here I am a bit confused with the paper which say the lr 
    # "decays by 0.97 every 2.4 epochs"
    # I use 0.97 as gamma and step size of 2
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                #step_size=cfg['scheduler_step_size'], 
                                                #gamma=cfg['scheduler_gamma'])

    scheduler = CosineScheduler(max_update=1000, 
                                base_lr=cfg['learning_rate'], 
                                final_lr=0.0000001, 
                                warmup_steps=5, 
                                warmup_begin_lr=0)

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None 

    for epoch in range(1, (cfg['epochs']+1)):

        if scheduler:
            if scheduler.__module__ == torch.optim.lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

        print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))

        train(train_dataloader, net, criterion, optimizer, epoch, device, scaler)
        validate(val_dataloader, net, criterion, epoch, device)
        
        
        
        if cfg['save_interval'] != 0 and epoch % cfg['save_interval'] == 0:
            torch.save(net.state_dict(), os.path.join(work_dir, 'epoch_{}.pth'.format(epoch)))

    torch.save(net.state_dict(), os.path.join(work_dir, 'final_{}.pth'.format(epoch)))


if __name__ == '__main__':
    main()
