
configs = {
    'efficientnet-b0': {
        'res': 224,
        'dropout': 0.2,
        'width_coef': 1.0, # coefficients to build the different effnets B0-B7
        'depth_coef': 1.0,
        'stages': [
            # expand_ratio, filter_size, num_repeats, in_channels, out_channels, stride, padding, se_ratio
            (1, 3, 1, 32, 16, 1, 'same', 0.25),
            (6, 3, 2, 16, 24, 2, 1, 0.25),
            (6, 5, 2, 24, 40, 2, 1, 0.25),
            (6, 3, 3, 40, 80, 2, 1, 0.25),
            (6, 5, 3, 80, 112, 1, 'same', 0.25),
            (6, 5, 4, 112, 192, 2, 1, 0.25),
            (6, 3, 1, 192, 320, 1, 'same', 0.25)
        ]

    }
}


def get_config(name):
    return configs[name]


def get_n_params(model):
    "Get the number of parameters of model"
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


