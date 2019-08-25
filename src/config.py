# -*- coding: utf-8 -*-


import warnings


class Config(object):

    
    log_dir = './src/logger/'
    data_dir = './data/ShanghaiTech/part_A/'
    model = 'CSRNet'
    model_path = './checkpoints/CSRNet/2019-08-19_09-41-28_10.pth'
    print_seq = 10
    lr = 1e-6
    lr_decay = 0.90
    weight_decay = 5*1e-4
    epochs = 100
    env = 'main'
    use_gpu = True

    def parse_args(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn("Invalid option `%s`" % key)
