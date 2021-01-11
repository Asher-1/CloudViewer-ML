#!/usr/bin/env python
import argparse
import math
import numpy as np
import os
import random
import pandas as pd
import sys
import torch
import argparse
import cloudViewer.ml as _ml3d
import cloudViewer.ml.torch as ml3d
from os.path import exists, join, isfile, dirname, abspath, split

DEVICE = "gpu:1"
ckpt_path = "/media/yons/data/develop/pcl_projects/ErowCloudViewer/runtimeDll/" \
            "CloudViewer_ML_Models/semantic_segmentation/randlanet_semantic3d_202012120312utc.pth"
data_dir = "/media/yons/data/dataset/pointCloud/data/Semantic3D/cloudViewer_processed"
cfg_file = "/media/yons/data/develop/pcl_projects/ErowCloudViewer/" \
           "CloudViewer-ML/ml3d/configs/randlanet_electricity3d.yml"


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for training and inference')
    parser.add_argument('--main_log_dir', help='the dir to save logs and models')
    parser.add_argument('--dataset_path', help='path to Electricity3D', default=data_dir)
    parser.add_argument('--split', help='train or test', default='test')
    parser.add_argument('--ckpt_path', help='model path', default=ckpt_path)
    parser.add_argument('--cfg_file', help='configure file', default=cfg_file)
    parser.add_argument('--device', default=DEVICE, help='path to RandLANet checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def get_custom_data(pc_names, path):
    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = os.path.join(path, name)
        pc = pd.read_csv(pc_path,
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values
        points = pc[:, 0:3]
        feats = pc[:, [4, 5, 6]]

        label_path = pc_path.replace(".txt", ".labels")
        labels = pd.read_csv(label_path,
                             header=None,
                             delim_whitespace=True,
                             dtype=np.int32).values
        labels = np.array(labels, dtype=np.int32).reshape((-1,))

        data = {
            'point': points,
            'feat': feats,
            'label': labels,
        }
        pc_data.append(data)

    return pc_data


def pred_custom_data(pc_names, pcs, pipeline):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]

        results = pipeline.run_inference(data)
        pred_label = (results['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label[0] = 0

        label = data['label']
        pts = data['point']

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "pred": pred_label,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_randla",
            "points": pts,
            "labels": pred_label,
        }
        vis_points.append(vis_d)

    return vis_points


# ------------------------------


def main(args):
    kitti_labels = ml3d.datasets.Semantic3D.get_label_to_names()
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(kitti_labels.keys()):
        lut.add_label(kitti_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    cfg = _ml3d.utils.Config.load_from_file(args.cfg_file)

    cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
        _ml3d.utils.Config.merge_cfg_file(cfg, args, {"model": {"ckpt_path": args.ckpt_path}})

    model = ml3d.models.RandLANet(**cfg_dict_model)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, **cfg_dict_pipeline)
    pipeline.load_ckpt(model.cfg.ckpt_path)

    pc_names = ["domfountain_station3_xyz_intensity_rgb.txt", "untermaederbrunnen_station3_xyz_intensity_rgb.txt"]
    pcs = get_custom_data(pc_names, args.dataset_path)
    pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline)

    v.visualize(pcs_with_pred)


if __name__ == "__main__":
    args = parse_args()
    main(args)
