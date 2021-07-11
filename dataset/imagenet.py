from torch.utils.data import Dataset

class ImageNet(Dataset):
    def __init__(self, root, annotations, is_train, transform=None):
        self.root = root
        self.annotations = annotations
        self.is_train = is_train
        self.transform = transform

    def __getitem__(self, idx):
        return 0

    def __len__(self):
        return len(self.annotations)
