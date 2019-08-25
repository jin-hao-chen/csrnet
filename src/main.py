#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)
import fire
import numpy as np
import datetime
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torchnet.meter as meter

from src.config import Config
from src import utils
from src.dataset import Shanghai


opts = Config()
device = t.device('cuda' if opts.use_gpu and t.cuda.is_available() else 'cpu')
model = utils.load_model(opts.model)().to(device)

# must be 1
batch_size = 1

def parse_args(func):
    def wrapper_fn(*args, **kwargs):
        opts.parse_args(**kwargs)
        ret = func(**kwargs)
        return ret
    return wrapper_fn

def echo(msg):
    os.system('echo %s' % msg)

@parse_args
def train(**kwargs):
    opts.parse_args(**kwargs)
    global model
    if opts.model_path:
        model.load(opts.model_path, debug=True)
    shanghai = Shanghai(opts.data_dir, train=False)
    dataloader = DataLoader(shanghai, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.SGD(model.parameters(), momentum=0.95, lr=opts.lr, weight_decay=opts.weight_decay)
    # vis = utils.Visualizer(env='main')
    loss_meter = meter.AverageValueMeter()
    for epoch in range(opts.epochs):
        loss_meter.reset()
        for i, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            start = datetime.datetime.now()
            data = data.to(device)
            labels = labels.to(device)
            predicted = model(data)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()
            end = datetime.datetime.now()

            loss_meter.add(utils.tensor2numpy(loss))
            # vis.plot_scalar('loss', utils.tensor2numpy(loss))
            # print('Iteration: %s, loss: %s, predicted_count: %s, ground_truth: %s, cost_time: %ss' 
            #     % (i + 1, 
            #         utils.tensor2numpy(loss), 
            #         utils.tensor2numpy(predicted).sum(), 
            #         utils.tensor2numpy(labels).sum(),
            #         (end - start).seconds))
             echo('Iteration: %s, loss: %s, predicted_count: %s, ground_truth: %s, cost_time: %ss' 
                % (i + 1, 
                    utils.tensor2numpy(loss), 
                    utils.tensor2numpy(predicted).sum(), 
                    utils.tensor2numpy(labels).sum(),
                    (end - start).seconds))
            if (i + 1) % opts.print_seq == 0:
                avg_loss = loss_meter.value()[0]
                # print('Epoch: %s, iteration: %s, avg_loss: %s' % (epoch + 1, i + 1, avg_loss))
                echo('Epoch: %s, iteration: %s, avg_loss: %s' % (epoch + 1, i + 1, avg_loss))
                # vis.plot_scalar('avg_loss', avg_loss)
                density_map = utils.tensor2numpy(predicted)
                density_map = np.reshape(density_map, (density_map.shape[2], density_map.shape[3]))
                density_map_label = utils.tensor2numpy(labels)
                density_map_label = np.reshape(density_map_label, (density_map_label.shape[2], density_map_label.shape[3]))
                # vis.plot_heatmap('density_map_label', density_map_label)
                # vis.plot_heatmap('avg_density_map', density_map)
        model.save('checkpoints/', epoch + 1, utils.tensor2numpy(loss))
        # print('----Save weights at epoch %s----' % (epoch + 1))
        echo('----Save weights at epoch %s----' % (epoch + 1))
        # lr = utils.adjust_lr(optimizer, epoch + 1, opts.lr, lr_decay=opts.lr_decay)
        # print('====Adjust lr: %s====' % lr)
        # print('Epoch: %s, loss: %s' % (epoch + 1, utils.tensor2numpy(loss)))
        echo('Epoch: %s, loss: %s' % (epoch + 1, utils.tensor2numpy(loss)))
        

@parse_args
def eval(**kwargs):
    opts.parse_args(**kwargs)
    print('eval')

def help():
    print('help')

def main():
    fire.Fire()


if __name__ == "__main__":
    main()
