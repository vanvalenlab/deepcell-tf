# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for backbone_utils"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from absl.testing import parameterized

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.platform import test

from deepcell.utils import backbone_utils


class TestBackboneUtils(test.TestCase, parameterized.TestCase):

    def test_get_featurenet_backbone(self):
        backbone = 'featurenet'
        input_shape = (256, 256, 3)
        inputs = Input(shape=input_shape)
        # with self.test_session(use_gpu=True):
        K.set_image_data_format('channels_last')
        out = backbone_utils.get_backbone(backbone, inputs, return_dict=True)
        assert isinstance(out, dict)
        assert all(k.startswith('C') for k in out)

        out = backbone_utils.get_backbone(backbone, inputs, return_dict=False)
        assert isinstance(out, Model)

        with self.assertRaises(ValueError):
            out = backbone_utils.get_backbone(
                backbone, inputs, use_imagenet=True)

    def test_get_featurenet3d_backbone(self):
        backbone = 'featurenet3d'
        input_shape = (40, 256, 256, 3)
        inputs = Input(shape=input_shape)
        # with self.test_session(use_gpu=True):
        K.set_image_data_format('channels_last')
        out = backbone_utils.get_backbone(backbone, inputs, return_dict=True)
        assert isinstance(out, dict)
        assert all(k.startswith('C') for k in out)

        out = backbone_utils.get_backbone(backbone, inputs, return_dict=False)
        assert isinstance(out, Model)

        with self.assertRaises(ValueError):
            out = backbone_utils.get_backbone(
                backbone, inputs, use_imagenet=True)

    @parameterized.named_parameters([
        ('resnet50',) * 2,
        ('resnet101',) * 2,
        ('resnet152',) * 2,
        ('resnet50v2',) * 2,
        ('resnet101v2',) * 2,
        ('resnet152v2',) * 2,
        ('resnext50',) * 2,
        ('resnext101',) * 2,
        ('vgg16',) * 2,
        ('vgg19',) * 2,
        ('densenet121',) * 2,
        ('densenet169',) * 2,
        ('densenet201',) * 2,
        ('mobilenet',) * 2,
        ('mobilenetv2',) * 2,
        ('nasnet_large',) * 2,
        ('nasnet_mobile',) * 2,
    ])
    def test_get_backbone(self, backbone):
        # with self.test_session(use_gpu=True):
        K.set_image_data_format('channels_last')
        inputs = Input(shape=(256, 256, 3))
        out = backbone_utils.get_backbone(
            backbone, inputs, return_dict=True)
        assert isinstance(out, dict)
        assert all(k.startswith('C') for k in out)

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
