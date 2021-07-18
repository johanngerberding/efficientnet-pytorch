import pickle 
import torch
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    def __init__(self, meta_filepath, root, imgs_pkl, labels_pkl, is_train, transform=None):
        self.classes = self._get_classes(meta_filepath)
        self.root = root
        self.imgs = self.load_pickle(imgs_pkl) 
        self.labels = self.load_pickle(labels_pkl)
        self.is_train = is_train
        self.transform = transform

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        label = torch.tensor(label)

        if self.transform: 
            img = self.transform(image=img)["image"]
        
        return img, label 

    def __len__(self):
        return len(self.labels)

    def _get_classes(self, meta_filepath):
        with open(meta_filepath, 'rb') as info:
            labels = pickle.load(info, encoding ='bytes')

        return [l.decode() for l in labels[b'label_names']]

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data



class CIFAR100(Dataset):
    def __init__(self, meta_filepath, root, imgs_pkl, labels_pkl, is_train, transform=None):
        self.classes = self._get_classes(meta_filepath)
        self.root = root
        self.imgs = self.load_pickle(imgs_pkl) 
        self.labels = self.load_pickle(labels_pkl)
        self.is_train = is_train
        self.transform = transform

    def __getitem__(self, idx):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError