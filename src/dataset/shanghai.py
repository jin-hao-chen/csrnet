# -*- coding: utf-8 -*-


import os
import scipy.io
import numpy as np
import cv2
import h5py
from PIL import Image
import torch as t
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T

from src.config import Config
from src import utils


opts = Config()


class Shanghai(Dataset):


    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.files = []
        if train:
            self.prepare_for_train()
        else:
            self.prepare_for_test()
        self.trainsforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, 'r') as hf:
            image = np.array(hf['image'])
            if len(image.shape) == 2:
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image)
            density = np.array(hf['density'])
        if self.train:
            if np.random.random() > 0.5:
                image.transpose(Image.FLIP_LEFT_RIGHT)
                density = np.fliplr(density)
            if np.random.random() > 0.5:
                image.transpose(Image.FLIP_TOP_BOTTOM)
                density = np.flipud(density)
            crop_size = (image.size[0] // 2, image.size[1] // 2)
            if np.random.random() > 0.5:
                x = int(np.random.random() * image.size[0] // 2)
                y = int(np.random.random() * image.size[1] // 2)
                image = image.crop((x, y, x + crop_size[0], y + crop_size[1]))
                density = density[y:y + crop_size[1], x:x + crop_size[0]]
        image = self.trainsforms(image)
        density =  cv2.resize(density, (density.shape[1] // 8, density.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64
        density = np.reshape(density, (-1, ) + density.shape)
        density = density.copy()
        return image, density
    
    def prepare_for_train(self):
        self._prepare_for_data('train')
    
    def prepare_for_test(self):
        self._prepare_for_data('test')

    def _prepare_for_data(self, name):
        parent_dir = os.path.join(self.root, name)
        self.files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
