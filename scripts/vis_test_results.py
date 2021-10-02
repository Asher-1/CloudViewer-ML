#!/usr/bin/env python
import cloudViewer.ml.torch as ml3d
import numpy as np
import os, sys, glob, pickle
import pandas as pd
from os.path import join
from pathlib import Path


def get_custom_data(pc_names, data_path, labels_path, extend=".txt"):
    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = join(data_path, name + extend)
        label_path = join(labels_path, name + '.labels')
        if not os.path.exists(pc_path):
            print("cannot find pointcloud data corresponding to {}".format(pc_path))
            continue

        pc = pd.read_csv(pc_path,
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values

        # points = pc[:, 0:3]
        # feats = pc[:, 3:6]
        points = pc[:, 0:3]
        feats = pc[:, [4, 5, 6]]

        points = np.array(points, dtype=np.float32)
        feats = np.array(feats, dtype=np.float32)
        # feats = feats / 255

        labels = pd.read_csv(label_path,
                             header=None,
                             delim_whitespace=True,
                             dtype=np.int32).values
        labels = np.array(labels, dtype=np.int32).reshape((-1,))

        data = {
            "name": name + "_randlanet",
            'points': points,
            'colors': feats,
            'labels': labels
        }

        pc_data.append(data)

    return pc_data


# ------------------------------


def main():
    EXTEND = ".txt"  # ".xyz"
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    label_path_name = "Semantic3D_torch"  # "Electricity3D_torch"
    test_label_path = os.path.join(CURRENT_PATH, "test", label_path_name)

    # data_path = "/media/yons/data/dataset/pointCloud/data/Electricity3D/cloudViewer_processed"
    # electricity_labels = ml3d.datasets.Electricity3D.get_label_to_names()

    data_path = "/media/yons/data/dataset/pointCloud/data/Semantic3D/cloudViewer_processed"
    data_labels = ml3d.datasets.Semantic3D.get_label_to_names()

    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(data_labels.keys()):
        lut.add_label(data_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    all_files = glob.glob(str(Path(test_label_path) / '*.labels'))

    pc_names = [os.path.basename(file).split(".")[0] for file in all_files]

    pcs_with_pred = get_custom_data(pc_names, data_path, test_label_path, extend=EXTEND)

    v.visualize(pcs_with_pred)


if __name__ == "__main__":
    main()
