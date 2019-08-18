# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase
import torch as t
from torch.utils.data.dataloader import DataLoader

from src.dataset import Shanghai
from src.config import Config


opts = Config()


class DatasetTest(TestCase):


    def test_dataset(self):
        shanghai = Shanghai(opts.data_dir, train=True)
        dataloader = DataLoader(shanghai, batch_size=1, shuffle=True, num_workers=2)
        for i, (data, labels) in enumerate(dataloader):
            print(i, labels[0].sum())