import math 

B0_STAGES = [
    # expand_ratio, filter_size, num_repeats, in_channels, out_channels, stride, padding, se_ratio
    (1, 3, 1, 32, 16, 1, 'same', 0.25),
    (6, 3, 2, 16, 24, 2, 1, 0.25),
    (6, 5, 2, 24, 40, 2, 1, 0.25),
    (6, 3, 3, 40, 80, 2, 1, 0.25),
    (6, 5, 3, 80, 112, 1, 'same', 0.25),
    (6, 5, 4, 112, 192, 2, 1, 0.25),
    (6, 3, 1, 192, 320, 1, 'same', 0.25)
]

CONFIGS = {
    'efficientnet-b0': {
        'res': 224,
        'dropout': 0.2,
        'width_coef': 1.0, # coefficients to build the different effnets B0-B7
        'depth_coef': 1.0,
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

class ModelParams(object):
    def __init__(self, name, num_classes=1000):
        super().__init__()
        self.name = name 
        self.num_classes = num_classes
        self.img_size = CONFIGS[self.name]['res']
        self.dropout = CONFIGS[self.name]['dropout']
        self.width_coef = CONFIGS[self.name]['width_coef']
        self.depth_coef = CONFIGS[self.name]['depth_coef']
        self.depth_divisor = 8 
        self.min_depth = None
        self.bn_momentum = 0.99
        self.bn_epsilon = 1e-3
        self.stages = self.get_stages(B0_STAGES)



    def calc_num_repeats(self, num_repeats):
        "New number of repeats"
        num_repeats = int(math.ceil(self.depth_coef * num_repeats))
        return num_repeats
    
    def num_channels(self, num_channels):
        filters = num_channels * self.width_coef
        min_depth = self.min_depth or self.depth_divisor
        new_filters = max(min_depth, int(filters + self.depth_divisor / 2) // self.depth_divisor * self.depth_divisor)
        if new_filters < 0.9 * filters:
            new_filters += self.depth_divisor
        return int(new_filters)

    def get_stages(self, baseline):
        stages = []
        for stage in baseline:
            num_repeats = self.calc_num_repeats(stage[2])
            num_in_channels = self.num_channels(stage[3])
            num_out_channels = self.num_channels(stage[4])
            new_stage = (stage[0], 
                         stage[1], 
                         num_repeats, 
                         num_in_channels, 
                         num_out_channels, 
                         stage[5], 
                         stage[6], 
                         stage[7])
            stages.append(new_stage)

        return stages


def get_config(name):
    return CONFIGS[name]


def get_n_params(model):
    "Get the number of parameters of model"
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

"""
def main():
    params = ModelParams('efficientnet-b6')
    for stage in params.stages:
        print(stage)


if __name__ == '__main__':
    main()"""