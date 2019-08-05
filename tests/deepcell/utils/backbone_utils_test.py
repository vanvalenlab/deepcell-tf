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
from __future__ import division
from __future__ import print_function

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
