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
"""Test the RetinaMask models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import keras_parameterized

from deepcell.model_zoo import PanopticNet


class PanopticNetTest(keras_parameterized.TestCase):

    # @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters([
        {
            'testcase_name': 'panopticnet_basic',
            'pooling': None,
            'location': False,
        },
        {
            'testcase_name': 'panopticnet_location',
            'pooling': None,
            'location': True,
        },
        {
            'testcase_name': 'panopticnet_avgpool',
            'pooling': 'avg',
            'location': False,
        },
        {
            'testcase_name': 'panopticnet_maxpool',
            'pooling': 'max',
            'location': False,
        },
        {
            'testcase_name': 'panopticnet_location_maxpool',
            'pooling': 'max',
            'location': True,
        },
        {
            'testcase_name': 'panopticnet_location_avgpool',
            'pooling': 'avg',
            'location': True,
        }
    ])
    def test_maskrcnn(self, pooling, location):
        num_classes = 3
        crop_size = (14, 14)
        mask_size = (28, 28)

        norm_method = None

        # not all backbones work with channels_first
        backbone = 'featurenet'

        # TODO: RetinaMask fails with channels_first
        data_format = 'channels_last'

        with self.cached_session():
            K.set_image_data_format(data_format)
            if data_format == 'channels_first':
                axis = 1
                input_shape = (1, 32, 32)
            else:
                axis = -1
                input_shape = (32, 32, 1)

            num_semantic_classes = [1, 3]

            model = PanopticNet(
                backbone=backbone,
                input_shape=input_shape,
                backbone_levels=['C3', 'C4', 'C5'],
                norm_method=norm_method,
                location=location,
                pooling=pooling,
                num_semantic_heads=len(num_semantic_classes),
                num_semantic_classes=num_semantic_classes,
                use_imagenet=False,
            )

            self.assertIsInstance(model.output_shape, list)
            self.assertEqual(len(model.output_shape), len(num_semantic_classes))
            for i, s in enumerate(num_semantic_classes):
                self.assertEqual(model.output_shape[i][axis], s)
