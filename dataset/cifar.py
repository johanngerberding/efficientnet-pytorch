import pickle 
import os 
from torch.utils.data import Dataset

class CIFAR(Dataset):
    def __init__(self, num_classes, root, annotations, is_train, transform=None):
        self.num_classes = num_classes
        self.root = root
        self.annotations = annotations
        self.is_train = is_train
        self.transform = transform


    def __getitem__(self, idx):
        return NotImplementedError

    def __len__(self):
        return len(self.annotations)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    cifar10 = "/home/johann/dev/efficientnet-pytorch/data/cifar10"
    #cifar100 = "/media/data/CIFAR/cifar-10-python"

    train_imgs = load_pickle(os.path.join(cifar10, 'train/train_imgs.pkl'))
    train_labels = load_pickle(os.path.join(cifar10, 'train/train_labels.pkl'))
    print(train_imgs.shape)
    print(train_labels.shape)



if __name__ == '__main__':
    main()