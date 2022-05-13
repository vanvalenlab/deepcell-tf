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
"""Test the RetinaMask models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.test import assert_equal_graph_def
from tensorflow.keras import backend as K
from tensorflow.python.keras import keras_parameterized

from deepcell.model_zoo import PanopticNet


class PanopticNetTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters([
        {
            'testcase_name': 'panopticnet_basic',
            'pooling': None,
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_location',
            'pooling': None,
            'location': True,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_avgpool',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_maxpool',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_basic_td',
            'pooling': None,
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_location_td',
            'pooling': None,
            'location': True,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_avgpool_td',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_maxpool_td',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_basic_cf',
            'pooling': None,
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_location_cf',
            'pooling': None,
            'location': True,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_avgpool_cf',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_maxpool_cf',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        },
        {
            'testcase_name': 'panopticnet_basic_td_cf',
            'pooling': None,
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_location_td_cf',
            'pooling': None,
            'location': True,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_avgpool_td_cf',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_maxpool_td_cf',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsamplelike',
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        },
        {
            'testcase_name': 'panopticnet_basic_upsampling2d',
            'pooling': None,
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_location_upsampling2d',
            'pooling': None,
            'location': True,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_avgpool_upsampling2d',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_maxpool_upsampling2d',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_basic_td_upsampling2d',
            'pooling': None,
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        },
        {
            'testcase_name': 'panopticnet_location_td_upsampling2d',
            'pooling': None,
            'location': True,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_avgpool_td_upsampling2d',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        },
        {
            'testcase_name': 'panopticnet_maxpool_td_upsampling2d',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_last',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_basic_cf_upsampling2d',
            'pooling': None,
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_location_cf_upsampling2d',
            'pooling': None,
            'location': True,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_avgpool_cf_upsampling2d',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_maxpool_cf_upsampling2d',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 1,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_basic_td_cf_upsampling2d',
            'pooling': None,
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_location_td_cf_upsampling2d',
            'pooling': None,
            'location': True,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        },
        {
            'testcase_name': 'panopticnet_avgpool_td_cf_upsampling2d',
            'pooling': 'avg',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'panopticnet_maxpool_td_cf_upsampling2d',
            'pooling': 'max',
            'location': False,
            'frames_per_batch': 3,
            'data_format': 'channels_first',
            'upsample_type': 'upsampling2d',
            'pyramid_levels': ['P3'],
        },

    ])
    def test_panopticnet(self, pooling, location, frames_per_batch,
                         data_format, upsample_type, pyramid_levels):
        norm_method = None

        # not all backbones work with channels_first
        backbone = 'featurenet'

        # TODO: PanopticNet fails with channels_first and frames_per_batch > 1
        if frames_per_batch > 1 and data_format == 'channels_first':
            return

        with self.cached_session():
            K.set_image_data_format(data_format)
            if data_format == 'channels_first':
                axis = 1
                input_shape = (1, 32, 32)
            else:
                axis = -1
                input_shape = (32, 32, 1)

            num_semantic_classes = [1, 3]

            # temporal_mode=None,
            # lite=False,
            # interpolation='bilinear',

            model = PanopticNet(
                backbone=backbone,
                input_shape=input_shape,
                frames_per_batch=frames_per_batch,
                pyramid_levels=pyramid_levels,
                norm_method=norm_method,
                location=location,
                pooling=pooling,
                upsample_type=upsample_type,
                num_semantic_classes=num_semantic_classes,
                use_imagenet=False,
            )

            self.assertIsInstance(model.output_shape, list)
            self.assertEqual(len(model.output_shape), len(num_semantic_classes))
            for i, s in enumerate(num_semantic_classes):
                self.assertEqual(model.output_shape[i][axis], s)

    @keras_parameterized.run_all_keras_modes
    def test_panopticnet_semantic_class_types(self):
        shared_kwargs = {
            'backbone': 'featurenet',
            'input_shape': (32, 32, 1),
            'use_imagenet': False,
        }

        with self.cached_session():
            nsc1 = [2, 3]
            model1 = PanopticNet(num_semantic_classes=nsc1, **shared_kwargs)

            nsc2 = {'0': 2, '1': 3}
            model2 = PanopticNet(num_semantic_classes=nsc1, **shared_kwargs)

            for o1, o2 in zip(model1.outputs, model2.outputs):
                self.assertEqual(o1.shape.as_list(), o2.shape.as_list())
                self.assertEqual(o1.name, o2.name)
                self.assertEqual(o1.dtype, o2.dtype)

    def test_panopticnet_bad_input(self):

        norm_method = None

        # not all backbones work with channels_first
        backbone = 'featurenet'

        num_semantic_classes = [1, 3]

        # non-square input
        input_shape = (256, 512, 1)
        with self.assertRaises(ValueError):
            model = PanopticNet(
                backbone=backbone,
                input_shape=input_shape,
                backbone_levels=['C3', 'C4', 'C5'],
                norm_method=norm_method,
                location=True,
                pooling='avg',
                num_semantic_classes=num_semantic_classes,
                use_imagenet=False,
            )

        # non power of 2 input
        input_shape = (257, 257, 1)
        with self.assertRaises(ValueError):
            model = PanopticNet(
                backbone=backbone,
                input_shape=input_shape,
                backbone_levels=['C3', 'C4', 'C5'],
                norm_method=norm_method,
                location=True,
                pooling='avg',
                num_semantic_classes=num_semantic_classes,
                use_imagenet=False,
            )
