# -*- coding: utf-8 -*-


import os
import time
import warnings
import numpy as np
import torch as t
import torch.nn as nn
import json


class BaseModule(nn.Module):


    def __init__(self):
        super(BaseModule, self).__init__()
        self.name = str(self.__class__).split("'")[1].split('.')[-1]
    
    def save(self, root, epoch, loss, max_items=5):
        if isinstance(loss, np.ndarray):
            loss = loss.tolist()
            if isinstance(loss, list):
                loss = loss[0]
        model_dir = os.path.join(root, self.name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        checkpoints_path = os.path.join(model_dir, 'checkpoints.json')
        if not os.path.exists(checkpoints_path):
            with open(checkpoints_path, 'w') as f:
                f.write('{}')
        with open(checkpoints_path, 'r') as f:
            data = json.load(f)
        if len(data) == max_items:
            data = sorted(data.items(), key=lambda x: x[1])
            name = data.pop()[0]
            full_name = os.path.join(model_dir, name)
            os.remove(full_name)
            data = dict(data)
            with open(checkpoints_path, 'w') as f:
                json.dump(data, f, indent=4)
        with open(checkpoints_path, 'r') as f:
            data = json.load(f)
        data = dict(sorted(data.items(), key=lambda x: x[1]))
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(model_dir, current_time + '_' + str(epoch) + '.pth')
        t.save(self.state_dict(), path)
        data[current_time + '_' + str(epoch) + '.pth'] = loss
        with open(checkpoints_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, path, debug=False):
        if not os.path.exists(path):
            warnings.warn("Path %s doesn't exist" % path)
        else:
            self.load_state_dict(t.load(path))
            if debug:
                print("Load weights from %s" % path)