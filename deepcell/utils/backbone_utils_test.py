# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras import keras_parameterized

from deepcell.utils import backbone_utils


class TestBackboneUtils(keras_parameterized.TestCase):

    @keras_parameterized.run_with_all_model_types
    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            data_format=[
                # 'channels_first',
                'channels_last']))
    def test_get_featurenet_backbone(self, data_format):
        backbone = 'featurenet'
        input_shape = (256, 256, 3)
        inputs = Input(shape=input_shape)
        with self.cached_session():
            K.set_image_data_format(data_format)
            model, output_dict = backbone_utils.get_backbone(
                backbone, inputs, return_dict=True)
            assert isinstance(output_dict, dict)
            assert all(k.startswith('C') for k in output_dict)
            assert isinstance(model, Model)

            # No imagenet weights for featurenet backbone
            with self.assertRaises(ValueError):
                backbone_utils.get_backbone(backbone, inputs, use_imagenet=True)

    # @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            data_format=[
                # 'channels_first',
                'channels_last']))
    def test_get_featurenet3d_backbone(self, data_format):
        backbone = 'featurenet3d'
        input_shape = (40, 256, 256, 3)
        inputs = Input(shape=input_shape)
        with self.cached_session():
            K.set_image_data_format(data_format)
            model, output_dict = backbone_utils.get_backbone(
                backbone, inputs, return_dict=True)
            assert isinstance(output_dict, dict)
            assert all(k.startswith('C') for k in output_dict)
            assert isinstance(model, Model)

            # No imagenet weights for featurenet backbone
            with self.assertRaises(ValueError):
                backbone_utils.get_backbone(backbone, inputs, use_imagenet=True)

    # @keras_parameterized.run_with_all_model_types
    # @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            backbone=[
                'resnet50',
                'resnet101',
                'resnet152',
                'resnet50v2',
                'resnet101v2',
                'resnet152v2',
                # 'resnext50',
                # 'resnext101',
                'vgg16',
                'vgg19',
                'densenet121',
                'densenet169',
                'densenet201',
                'mobilenet',
                'mobilenetv2',
                'efficientnetb0',
                'efficientnetb1',
                'efficientnetb2',
                'efficientnetb3',
                'efficientnetb4',
                'efficientnetb5',
                'efficientnetb6',
                'efficientnetb7',
                'nasnet_large',
                'nasnet_mobile']))
    def test_get_backbone(self, backbone):
        with self.cached_session():
            K.set_image_data_format('channels_last')
            inputs = Input(shape=(256, 256, 3))
            model, output_dict = backbone_utils.get_backbone(
                backbone, inputs, return_dict=True)
            assert isinstance(output_dict, dict)
            assert all(k.startswith('C') for k in output_dict)
            assert isinstance(model, Model)

    def test_invalid_backbone(self):
        inputs = Input(shape=(4, 2, 3))
        with self.assertRaises(ValueError):
            backbone_utils.get_backbone('bad', inputs, return_dict=True)


if __name__ == '__main__':
    test.main()
