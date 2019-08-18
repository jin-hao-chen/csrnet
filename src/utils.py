# -*- coding: utf-8 -*-


import os
import time
import warnings
import scipy
import scipy.io as io
import numpy as np
import torch as t
import visdom
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial

from src.config import Config
from src import models


def pil2cv(image):
    """Convert pillow image to cv image
    """
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def cv2pil(image):
    """Convert cv image to pillow image
    """
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def cv2array(image):
    """Convert ndarray type image format from BGR to RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def pil2array(image):
    """Convert pillow image to ndarray
    """
    return np.array(image)

def tensor2numpy(tensor):
    """Convert tensor to ndarray
    """
    return tensor.cpu().detach().numpy()

def array2tensor(array, device='auto'):
    """Convert ndarray to tensor on ['cpu', 'gpu', 'auto']
    """
    assert device in ['cpu', 'gpu', 'auto'], "Invalid device"
    if device != 'auto':
        return t.tensor(array).float().to(t.device(device))
    if device == 'auto':
        return t.tensor(array).float().to(t.device('cuda' if t.cuda.is_available() else 'cpu'))

def adjust_lr(optimizer, epoch, initial_lr, lr_decay=0.95):
    """Classical method to adjust learning rate
    """
    lr = initial_lr / (1.0 + lr_decay * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def load_mat(path):
    return io.loadmat(path)

# Reference from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2.0 / 2.0 # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return np.array(density)

def show_density(density):
    plt.imshow(density, cmap=CM.jet)
    plt.show()

class Visualizer(object):


    def __init__(self, env='main'):
        self.vis = visdom.Visdom(env=env)
        self.index = {}
    
    def plot_scalar(self, name, y):
        """
        Parameters
        ----------
        name : str

        y : array-like but not torch.tensor
        """
        if name not in self.index:
            self.index[name] = 0
        x = self.index[name]
        self.vis.line(Y=[np.array(y)], X=[np.array(x)], win=name, update='append', name=name)
        self.index[name] += 1
    
    def plot_image(self, name, img):
        """
        Parameters
        ----------
        name : str

        img : array-like but not torch.tensor
        """
        self.vis.image(np.array(img), win=name)
    
    def plot_heatmap(self, name, img):
        self.vis.heatmap(img, win=name)


class Logger(object):


    def __init__(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.root = directory
    
    def log(self, name, s, overwrite=True):
        """
        Parameters
        ----------
        name : str
            log file name
        
        s : str
            content
        """
        path = os.path.join(self.root, name)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(path, 'w' if overwrite else 'a') as f:
            f.write('[%s]: %s\n' % (current_time, s))


def load_model(name):
    model = None
    if not hasattr(models, name):
        warnings.warn("Model `%s` doesn't exist in your defined model set, \
            please check src/models/__init__.py" % name)
    else:
        model = getattr(models, name)
    return model


logger = Logger(Config.log_dir)
