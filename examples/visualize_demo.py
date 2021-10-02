#!/usr/bin/env python
import cloudViewer.ml.torch as ml3d
import argparse
import math
import numpy as np
import os
import random
import sys
import torch
from os.path import exists, join, isfile, dirname, abspath, split


def get_custom_data(pc_names, path):
    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = join(path, 'points', name + '.npy')
        label_path = join(path, 'labels', name + '.npy')
        point = np.load(pc_path)[:, 0:3]
        label = np.squeeze(np.load(label_path))

        data = {
            'point': point,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)

    return pc_data


def pred_custom_data():

    num_points = 100000
    points = np.random.rand(num_points, 3).astype(np.float32)

    data = [
        {
            'name': 'my_point_cloud',
            'points': points,
            'random_colors': np.random.rand(*points.shape).astype(np.float32),
            'int_attr': (points[:, 0] * 5).astype(np.int32),
        }
    ]

    vis = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    lut.add_label('zero', 0)
    lut.add_label('one', 1)
    lut.add_label('two', 2)
    lut.add_label('three', 3, [0, 0, 1])  # use blue for label 'three'
    lut.add_label('four', 4, [0, 1, 0])  # use green for label 'four'
    vis.set_lut("int_attr", lut=lut)
    vis.visualize(data)


# ------------------------------


def main():
    pred_custom_data()


if __name__ == "__main__":
    main()
