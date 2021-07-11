from efficientnet.model import EfficientNet
from efficientnet.utils import get_n_params, get_config


def main():
    # training parameters
    batch_size = 8
    learning_rate = 0.01
    momentum = 0.9
    bn_momentum = 0.99
    decay = 0.9
    weight_decay = 0.00001

    config = get_config('efficientnet-b0')
    net = EfficientNet(config=config, num_classes=1000)

    print(net)
    num_params = get_n_params(net)
    print(num_params)

if __name__ == '__main__':
    main()
