from torch.utils.data import Dataset

class CIFAR(Dataset):
    def __init__(self, num_classes, root, annotations, is_train, transform=None):
        self.num_classes = num_classes
        self.root = root
        self.annotations = annotations
        self.is_train = is_train
        selt.transform = transform


    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return len(self.annotations)

