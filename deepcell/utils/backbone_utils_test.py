import os
import json
import tarfile
import tempfile

from absl.testing import parameterized

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from deepcell.utils import backbone_utils


class TestBackboneUtils(test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters([
        # ('resnet50',) * 2,
        ('featurenet',) * 2,
        ('featurenet3d',) * 2,
        # ('vgg16',) * 2,
        # ('vgg19',) * 2,
        # ('densenet121',) * 2,
        # ('densenet169',) * 2,
        # ('densenet201',) * 2,
        # ('mobilenet',) * 2,
        # ('mobilenetv2',) * 2,
        # ('mobilenet_v2',) * 2,
        # ('nasnet_large',) * 2,
        # ('nasnet_mobile',) * 2,
    ])
    def test_get_backbone(self, backbone):
        inputs = Input(shape=(4, 2, 3))
        out = backbone_utils.get_backbone(
            backbone, inputs, return_dict=True)
        assert isinstance(out, dict)
        assert 'C1' in out
        out = backbone_utils.get_backbone(
            backbone, inputs, return_dict=False)
        assert isinstance(out, Model)

    def test_invalid_backbone(self):
        inputs = Input(shape=(4, 2, 3))
        with self.assertRaises(ValueError):
            backbone_utils.get_backbone('bad', inputs, return_dict=True)
        with self.assertRaises(ValueError):
            backbone_utils.get_backbone('bad', inputs, return_dict=False)


if __name__ == '__main__':
    test.main()
