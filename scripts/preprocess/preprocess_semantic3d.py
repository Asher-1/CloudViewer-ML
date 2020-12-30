import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from tqdm import tqdm
import random
import shutil
from tqdm import tqdm
import argparse
from cloudViewer.ml.datasets import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split large pointclouds in Semantic3D.')
    parser.add_argument('--dataset_path',
                        help='path to Semantic3D',
                        required=True)
    parser.add_argument('--out_path', help='Output path', default=None)

    parser.add_argument(
        '--size_limit',
        help='Maximum size of processed pointcloud in Megabytes.',
        default=2000,
        type=int)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def preprocess(args):
    # Split large pointclouds into multiple point clouds.

    extend = ".txt"
    label_extend = ".labels"

    dataset_path = args.dataset_path
    out_path = args.out_path
    size_limit = args.size_limit  # Size in mega bytes.

    if out_path is None:
        out_path = Path(dataset_path) / 'processed'
        print("out_path not give, Saving output in {}".format(out_path))

    all_files = glob.glob(str(Path(dataset_path) / '*.txt'))

    train_files = [
        f for f in all_files
        if exists(str(Path(f).parent / Path(f).name.replace(extend, label_extend)))
    ]

    test_files = [
        f for f in all_files if f not in train_files
    ]

    files = {}
    for f in train_files:
        size = Path(f).stat().st_size / 1e6
        if size <= size_limit:
            files[f] = 1
            continue
        else:
            parts = int(size / size_limit) + 1
            files[f] = parts

    os.makedirs(out_path, exist_ok=True)

    sub_grid_size = 0.01

    # for train files
    print("preprocess train files......")
    for key, parts in tqdm(files.items()):
        # check if it has already preprocessed
        if exists(join(out_path, Path(key).name.replace(extend, extend))):
            print("ignore train file {} because already preprocessed, "
                  "you should remove it before running!".format(Path(key).name.replace(extend, extend)))
            continue

        if parts == 1:
            if dataset_path != out_path:
                pc = pd.read_csv(key,
                                 header=None,
                                 delim_whitespace=True,
                                 dtype=np.float32).values

                labels = pd.read_csv(key.replace(extend, label_extend),
                                     header=None,
                                     delim_whitespace=True,
                                     dtype=np.int32).values
                labels = np.array(labels, dtype=np.int32).reshape((-1,))

                raw_pc_shape = pc.shape
                raw_labels_shape = labels.shape
                points, feat, labels = utils.DataProcessing.grid_subsampling(
                    pc[:, :3],
                    features=pc[:, 3:],
                    labels=labels,
                    grid_size=sub_grid_size)
                pc = np.concatenate([points, feat], 1)
                subsampling_pc_shape = pc.shape
                subsampling_labels_shape = labels.shape
                print("subsampling {} from {} to {}".format(Path(key).name.replace(extend, extend),
                                                            raw_pc_shape, subsampling_pc_shape))
                print("subsampling {} from {} to {}".format(Path(key).name.replace(extend, label_extend),
                                                            raw_labels_shape, subsampling_labels_shape))

                name = join(out_path, Path(key).name.replace(extend, extend))
                name_lbl = name.replace(extend, label_extend)

                np.savetxt(name, pc, fmt='%.3f %.3f %.3f %i %i %i %i')
                np.savetxt(name_lbl, labels, fmt='%i')

            continue
        print("Splitting {} into {} parts".format(Path(key).name, parts))
        pc = pd.read_csv(key,
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values

        labels = pd.read_csv(key.replace(extend, label_extend),
                             header=None,
                             delim_whitespace=True,
                             dtype=np.int32).values
        labels = np.array(labels, dtype=np.int32).reshape((-1,))

        axis = 1  # Longest axis.

        inds = pc[:, axis].argsort()
        pc = pc[inds]
        labels = labels[inds]
        pcs = np.array_split(pc, parts)
        lbls = np.array_split(labels, parts)

        for i, pc in enumerate(pcs):
            lbl = lbls[i]
            raw_pc_shape = pc.shape
            raw_labels_shape = lbl.shape
            pc, feat, lbl = utils.DataProcessing.grid_subsampling(
                pc[:, :3],
                features=pc[:, 3:],
                labels=lbl,
                grid_size=sub_grid_size)
            pcs[i] = np.concatenate([pc, feat], 1)
            lbls[i] = lbl

            subsampling_pc_shape = pc.shape
            subsampling_labels_shape = lbl.shape
            print("subsampling {} from {} to {}".format(Path(key).name.replace(extend, extend),
                                                        raw_pc_shape, subsampling_pc_shape))
            print("subsampling {} from {} to {}".format(Path(key).name.replace(extend, label_extend),
                                                        raw_labels_shape, subsampling_labels_shape))

        for i in range(parts):
            name = join(
                out_path,
                Path(key).name.replace(extend, '_part_{}.txt'.format(i)))
            name_lbl = name.replace(extend, label_extend)

            shuf = np.arange(pcs[i].shape[0])
            np.random.shuffle(shuf)

            np.savetxt(name, pcs[i][shuf], fmt='%.3f %.3f %.3f %i %i %i %i')
            np.savetxt(name_lbl, lbls[i][shuf], fmt='%i')

    # for test files
    print("preprocess test files......")
    for key in tqdm(test_files):
        # check if it has already preprocessed
        if exists(join(out_path, Path(key).name.replace(extend, extend))):
            print("ignore test file {} because already preprocessed, "
                  "you should remove it before running!".format(Path(key).name.replace(extend, extend)))
            continue

        pc = pd.read_csv(key,
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values
        raw_pc_shape = pc.shape
        points, feat = utils.DataProcessing.grid_subsampling(
            pc[:, :3],
            features=pc[:, 3:],
            grid_size=sub_grid_size)
        pc = np.concatenate([points, feat], 1)
        subsampling_pc_shape = pc.shape
        print("subsampling {} from {} to {}".format(Path(key).name.replace(extend, extend),
                                                    raw_pc_shape, subsampling_pc_shape))

        name = join(out_path, Path(key).name.replace(extend, extend))
        np.savetxt(name, pc, fmt='%.3f %.3f %.3f %i %i %i %i')


if __name__ == '__main__':
    args = parse_args()
    preprocess(args)
