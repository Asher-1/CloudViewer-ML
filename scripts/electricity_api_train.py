import os
os.environ['CLOUDVIEWER_ML_ROOT'] = "/media/yons/data/develop/pcl_projects/ErowCloudViewer/CloudViewer-ML"

from cloudViewer.ml.datasets import (SemanticKITTI, ParisLille3D, Semantic3D, S3DIS, Toronto3D, Electricity3D)
from cloudViewer.ml.torch.pipelines import SemanticSegmentation
from cloudViewer.ml.torch.models import RandLANet
from cloudViewer.ml.utils import Config, get_module

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for training and inference')
    parser.add_argument('--path_electricity3d',
                        help='path to Electricity3D',
                        required=True)
    parser.add_argument('--path_ckpt_randlanet',
                        help='path to RandLANet checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def demo_train(args):
    # Initialize the training by passing parameters
    dataset = Electricity3D(args.path_electricity3d, use_cache=True)

    model = RandLANet(num_layers=5, num_points=65536, num_classes=19,
                      sub_sampling_ratio=[4, 4, 4, 4, 2], dim_input=6,
                      dim_output=[16, 64, 128, 256, 512], grid_size=0.03)

    pipeline = SemanticSegmentation(model=model,
                                    dataset=dataset,
                                    batch_size=2,
                                    val_batch_size=1,
                                    test_batch_size=1,
                                    max_epoch=100,  # maximum epoch during training
                                    learning_rate=1e-2,  # initial learning rate
                                    save_ckpt_freq=5)

    pipeline.run_train()


def demo_inference(args):
    # Inference and test example
    from cloudViewer.ml.tf.pipelines import SemanticSegmentation
    from cloudViewer.ml.tf.models import RandLANet

    Pipeline = get_module("pipeline", "SemanticSegmentation", "tf")
    Model = get_module("model", "RandLANet", "tf")
    Dataset = get_module("dataset", "Electricity3D")

    RandLANet = Model(num_layers=5, num_points=65536, num_classes=19,
                      sub_sampling_ratio=[4, 4, 4, 4, 2], dim_input=6,
                      dim_output=[16, 64, 128, 256, 512], grid_size=0.03,
                      ckpt_path=args.path_ckpt_randlanet)

    # Initialize by specifying config file path
    Electricity3D = Dataset(args.path_electricity3d, use_cache=False)

    pipeline = Pipeline(model=RandLANet, dataset=Electricity3D)

    # inference
    # get data
    train_split = Electricity3D.get_split("train")
    data = train_split.get_data(0)
    # restore weights

    # run inference
    results = pipeline.run_inference(data)
    print(results)

    # test
    pipeline.run_test()


if __name__ == '__main__':
    args = parse_args()
    demo_train(args)
    demo_inference(args)
