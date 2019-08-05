import os
import json
import tarfile
import tempfile

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from deepcell.utils import backbone_utils


class TestBackboneUtils(test.TestCase):

    def test_get_backbone(self):
        inputs = Input(shape=(4, 2, 3))
        valid_backbones = [
            'resnet50',
            'featurenet', 'featurenet3d', 'featurenet_3d',
            'vgg16', 'vgg19',
            'densenet121', 'densenet169', 'densenet201',
            'mobilenet', 'mobilenetv2', 'mobilenet_v2',
            'nasnet_large', 'nasnet_mobile',
        ]
        for backbone in valid_backbones:
            out = backbone_utils.get_backbone(
                backbone, inputs, return_dict=True)
            assert isinstance(out, dict)
            assert 'C1' in out
            out = backbone_utils.get_backbone(
                backbone, inputs, return_dict=False)
            assert isinstance(out, Model)

        with self.assertRaises(ValueError):
            backbone_utils.get_backbone('bad', inputs, return_dict=True)


if __name__ == '__main__':
    test.main()
