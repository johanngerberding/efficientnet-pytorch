import os 
from efficientnet.model import EfficientNet
from efficientnet.utils import get_n_params, get_config
from dataset.cifar import CIFAR10


def main():
    # training parameters
    batch_size = 8
    learning_rate = 0.01
    momentum = 0.9
    bn_momentum = 0.99
    decay = 0.9
    weight_decay = 0.00001

    config = get_config('efficientnet-b0')
    net = EfficientNet(config=config, num_classes=10)

    print(net)
    num_params = get_n_params(net)
    print(num_params)

    cifar10_root = "/home/johann/dev/efficientnet-pytorch/data/cifar10"
    meta_path = '/media/data/CIFAR/cifar-10-batches-py/batches.meta'
    train_imgs_pkl = os.path.join(cifar10_root, 'train/train_imgs.pkl')
    train_labels_pkl = os.path.join(cifar10_root, 'train/train_labels.pkl')

    dataset = CIFAR10(meta_path, cifar10_root, train_imgs_pkl, train_labels_pkl, True, None)
    print(len(dataset))


if __name__ == '__main__':
    main()
