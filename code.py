#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import h5py


def combine_hdf5(directory, target_dir):
    target = os.path.join(target_dir, 'data.h5')
    with h5py.File(target, 'w') as hfw:
        for fname in sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0])):
            prefix = fname.split('.')[0]
            path = os.path.join(directory, fname)
            with h5py.File(path, 'r') as hfr:
                group = hfw.create_group(prefix)
                group['image'] = np.array(hfr['image'])
                group['density'] = np.array(hfr['density'])
                print('Finish', prefix)
    print('Finish', target)

def main():
    combine_hdf5('./data/ShanghaiTech/part_B/test', './partB_test/')


if __name__ == "__main__":
    main()