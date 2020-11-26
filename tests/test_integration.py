import pytest
import os

if 'PATH_TO_OPEN3D_ML' in os.environ.keys():
    base = os.environ['PATH_TO_CLOUDVIEWER_ML']
else:
    base = '.'
    # base = '../CloudViewer-ML'


def test_integration_torch():
    import torch
    import cloudViewer.ml.torch as ml3d
    from cloudViewer.ml.datasets import S3DIS
    from cloudViewer.ml.utils import Config, get_module
    from cloudViewer.ml.torch.models import RandLANet, KPFCNN
    from cloudViewer.ml.torch.pipelines import SemanticSegmentation
    print(dir(ml3d))

    config = base + '/ml3d/configs/randlanet_toronto3d.yml'
    cfg = Config.load_from_file(config)

    model = ml3d.models.RandLANet(**cfg.model)

    print(model)


def test_integration_tf():
    import tensorflow as tf
    import cloudViewer.ml.tf as ml3d
    from cloudViewer.ml.datasets import S3DIS
    from cloudViewer.ml.utils import Config, get_module
    from cloudViewer.ml.tf.models import RandLANet, KPFCNN
    from cloudViewer.ml.tf.pipelines import SemanticSegmentation
    print(dir(ml3d))

    config = base + '/ml3d/configs/randlanet_toronto3d.yml'
    cfg = Config.load_from_file(config)

    model = ml3d.models.RandLANet(**cfg.model)

    print(model)


test_integration_torch()
