

stages = [
    # expand_ratio, filter_size, num_repeats, in_channels, out_channels, stride, padding, se_ratio
    (1, 3, 1, 32, 16, 1, 'same', 0.25),
    (6, 3, 2, 16, 24, 2, 1, 0.25),
    (6, 5, 2, 24, 40, 2, 1, 0.25),
    (6, 3, 3, 40, 80, 2, 1, 0.25),
    (6, 5, 3, 80, 112, 1, 'same', 0.25),
    (6, 5, 4, 112, 192, 2, 1, 0.25),
    (6, 3, 1, 192, 320, 1, 'same', 0.25)
]


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
    },
    'efficientnet-b1': {
        'res': 240,
        'dropout': 0.2,
        'width_coef': 1.0,
        'depth_coef': 1.1
    },
    'efficientnet-b2': {
        'res': 260,
        'dropout': 0.3,
        'width_coef': 1.1,
        'depth_coef': 1.2
    },
    'efficientnet-b3': {
        'res': 300,
        'dropout': 0.3,
        'width_coef': 1.2,
        'depth_coef': 1.4
    },
    'efficientnet-b4': {
        'res': 380,
        'dropout': 0.4,
        'width_coef': 1.4,
        'depth_coef': 1.8
    },
    'efficientnet-b5': {
        'res': 456,
        'dropout': 0.4,
        'width_coef': 1.6,
        'depth_coef': 2.2
    },
    'efficientnet-b6': {
        'res': 528,
        'dropout': 0.5,
        'width_coef': 1.8,
        'depth_coef': 2.6
    },
    'efficientnet-b7': {
        'res': 600,
        'dropout': 0.5,
        'width_coef': 2.0,
        'depth_coef': 3.1
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


