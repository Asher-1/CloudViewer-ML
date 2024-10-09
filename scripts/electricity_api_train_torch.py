import sys
import pprint
from pathlib import Path
import os

os.environ['CLOUDVIEWER_ML_ROOT'] = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import cloudViewer.ml as _ml3d
from cloudViewer.ml.utils import Config, get_module

import argparse

DATASET_NAME = "Semantic3D"

if DATASET_NAME == "Electricity3D":
    CKPT_PATH = None
    # CKPT_PATH = "/media/yons/data/develop/pcl_projects/ACloudViewer/CloudViewer-ML" \
    #             "/scripts/logs/RandLANet_Electricity3D_torch/checkpoint/ckpt_00100.pth"
    DATASET_PATH = "/media/yons/data/dataset/pointCloud/data/Electricity3D/cloudViewer_processed"
    CONFIG_FILE = "/media/yons/data/develop/pcl_projects/ACloudViewer/CloudViewer-ML/" \
                  "scripts/test_randlanet_electricity3d.yml"
else:
    CKPT_PATH = None
    # CKPT_PATH = "/media/yons/data/develop/pcl_projects/ACloudViewer/CloudViewer-ML/scripts/logs/" \
    #             "RandLANet_Semantic3D_torch/checkpoint/ckpt_00100.pth"
    # CKPT_PATH = "/media/yons/data/develop/pcl_projects/ACloudViewer/runtimeDll/randlanet_semantic3d_202012120312utc.pth"
    DATASET_PATH = "/media/yons/data/dataset/pointCloud/data/Semantic3D/cloudViewer_processed"
    CONFIG_FILE = "/media/yons/data/develop/pcl_projects/ACloudViewer/CloudViewer-ML/" \
                  "ml3d/configs/randlanet_semantic3d.yml"

DEVICE = "gpu:1"
FRAMEWORK = "torch"  # "torch", "tf"
BATCH_SIZE = 2
STEPS_PER_EPOCH_TRAIN = 500
GRID_SIZE = 0.06


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for training and inference')
    parser.add_argument('--dataset_path', help='path to Electricity3D', default=DATASET_PATH)
    parser.add_argument('--main_log_dir', help='the dir to save logs and models')
    parser.add_argument('--cfg_file', help='path to the config file', default=CONFIG_FILE)
    parser.add_argument('--batch_size', help='training batch size', default=BATCH_SIZE)
    parser.add_argument('--ckpt_path', default=CKPT_PATH, help='path to RandLANet checkpoint')
    parser.add_argument('--split', help='train or test', default='train')
    parser.add_argument('--framework', default=FRAMEWORK, help='path to RandLANet checkpoint')
    parser.add_argument('--device', default=DEVICE, help='path to RandLANet checkpoint')
    parser.add_argument('--steps_per_epoch_train', default=STEPS_PER_EPOCH_TRAIN,
                        help='steps per epoch train')
    parser.add_argument('--grid_size', default=GRID_SIZE, help='path to RandLANet checkpoint')


    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def demo_train(args):
    cmd_line = ' '.join(sys.argv[:])
    framework = _ml3d.utils.convert_framework_name(args.framework)
    if ":" in args.device:
        device_type = _ml3d.utils.convert_device_name(args.device.split(':')[0])
        idx = args.device.split(':')[1]
        args.device = "{}:{}".format(device_type, idx)
    else:
        args.device = _ml3d.utils.convert_device_name(args.device)

    if framework == 'torch':
        import cloudViewer.ml.torch as ml3d
    else:
        import tensorflow as tf
        import cloudViewer.ml.tf as ml3d

        device = args.device
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if device == 'cpu':
                    tf.config.set_visible_devices([], 'GPU')
                elif device == 'gpu':
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                else:
                    idx = device.split(':')[1]
                    tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
            except RuntimeError as e:
                print(e)

    if args.cfg_file is not None:
        cfg = _ml3d.utils.Config.load_from_file(args.cfg_file)

        Pipeline = _ml3d.utils.get_module("pipeline", cfg.pipeline.name, framework)
        Model = _ml3d.utils.get_module("model", cfg.model.name, framework)
        Dataset = _ml3d.utils.get_module("dataset", cfg.dataset.name)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
            _ml3d.utils.Config.merge_cfg_file(cfg, args,
                                              {"pipeline": {"batch_size": str(args.batch_size)},
                                               "model": {"grid_size": str(args.grid_size)},
                                               "dataset": {"steps_per_epoch_train": str(args.steps_per_epoch_train)},
                                               })

        dataset = Dataset(cfg_dict_dataset.pop('dataset_path', None), **cfg_dict_dataset)
        model = Model(**cfg_dict_model)
        pipeline = Pipeline(model, dataset, **cfg_dict_pipeline)
    else:
        Pipeline = get_module("pipeline", "SemanticSegmentation", framework)
        Model = get_module("model", "RandLANet", framework)
        Dataset = get_module("dataset", DATASET_NAME)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
            _ml3d.utils.Config.merge_module_cfg_file(args, {})

        # Initialize the training by passing parameters
        dataset = Dataset(args.ckpt_path, use_cache=True)

        model = Model(num_layers=5, num_points=65536, num_classes=19,
                      sub_sampling_ratio=[4, 4, 4, 4, 2], dim_input=6,
                      dim_output=[16, 64, 128, 256, 512], grid_size=GRID_SIZE)

        pipeline = Pipeline(model=model,
                            dataset=dataset,
                            batch_size=2,
                            val_batch_size=1,
                            test_batch_size=1,
                            max_epoch=100,  # maximum epoch during training
                            learning_rate=1e-2,  # initial learning rate
                            save_ckpt_freq=5)

    with open(Path(__file__).parent / 'README.md', 'r') as f:
        readme = f.read()
    pipeline.cfg_tb = {
        'readme': readme,
        'cmd_line': cmd_line,
        'dataset': pprint.pformat(cfg_dict_dataset, indent=2),
        'model': pprint.pformat(cfg_dict_model, indent=2),
        'pipeline': pprint.pformat(cfg_dict_pipeline, indent=2)
    }

    pipeline.run_train()


def demo_inference(args):
    framework = _ml3d.utils.convert_framework_name(args.framework)
    if ":" in args.device:
        device_type = _ml3d.utils.convert_device_name(args.device.split(':')[0])
        idx = args.device.split(':')[1]
        args.device = "{}:{}".format(device_type, idx)
    else:
        args.device = _ml3d.utils.convert_device_name(args.device)

    if framework == 'torch':
        import cloudViewer.ml.torch as ml3d
    else:
        import tensorflow as tf
        import cloudViewer.ml.tf as ml3d

        device = args.device
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if device == 'cpu':
                    tf.config.set_visible_devices([], 'GPU')
                elif device == 'gpu':
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                else:
                    idx = device.split(':')[1]
                    tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
            except RuntimeError as e:
                print(e)

    Pipeline = get_module("pipeline", "SemanticSegmentation", framework)
    Model = get_module("model", "RandLANet", framework)
    Dataset = get_module("dataset", DATASET_NAME)

    # Initialize by specifying config file path
    dataset = Dataset(args.dataset_path, use_cache=False)

    num_classes = dataset.num_classes - len(dataset.ignored_labels)
    RandLANet = Model(num_layers=5, num_points=65536, num_classes=num_classes,
                      sub_sampling_ratio=[4, 4, 4, 4, 2], dim_input=6,
                      dim_output=[16, 64, 128, 256, 512], grid_size=GRID_SIZE,
                      ckpt_path=args.ckpt_path)

    pipeline = Pipeline(model=RandLANet, dataset=dataset)

    # restore weights
    pipeline.load_ckpt(RandLANet.cfg.ckpt_path)

    # get data
    # train_split = dataset.get_split("train")
    # data = train_split.get_data(0)

    # run inference
    # results = pipeline.run_inference(data)
    # print(results)

    # test
    pipeline.run_test()


if __name__ == '__main__':
    args = parse_args()
    demo_train(args)
    demo_inference(args)
