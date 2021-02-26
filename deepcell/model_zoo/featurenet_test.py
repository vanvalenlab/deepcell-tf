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
"""Test the featurenet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized

from tensorflow.keras import backend as K

from deepcell.model_zoo import featurenet


class FeatureNetTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters([
        {
            'testcase_name': 'dilated_include_top',
            'include_top': True,
            'dilated': True,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (32, 32, 1),
            'data_format': 'channels_last'
        },
        {
            'testcase_name': 'dilated_include_top_cf',
            'include_top': True,
            'dilated': True,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (32, 32, 1),
            'data_format': 'channels_first'
        },
        # {
        #     'testcase_name': 'not_dilated_include_top',
        #     'include_top': True,
        #     'dilated': False,
        #     'padding_mode': 'reflect',
        #     'padding': True,
        #     'multires': False,
        #     'location': False,
        #     'shape': (32, 32, 1),
        #     'data_format': 'channels_last',
        # },
        # {
        #     'testcase_name': 'not_dilated_include_top_cf',
        #     'include_top': True,
        #     'dilated': False,
        #     'padding_mode': 'reflect',
        #     'padding': True,
        #     'multires': False,
        #     'location': False,
        #     'shape': (32, 32, 1),
        #     'data_format': 'channels_first',
        # },
        {
            'testcase_name': 'dilated_no_top',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (33, 33, 1),
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'dilated_no_top_cf',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (33, 33, 1),
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'not_dilated_no_top',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (32, 32, 1),
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'not_dilated_no_top_cf',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (32, 32, 1),
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'not_dilated_no_top_location',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': True,
            'shape': (32, 32, 1),
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'not_dilated_no_top_location_cf',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': True,
            'shape': (32, 32, 1),
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'not_dilated_no_top_multires',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': False,
            'shape': (32, 32, 1),
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'not_dilated_no_top_multires_cf',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': False,
            'shape': (32, 32, 1),
            'data_format': 'channels_first',
        },
    ])
    def test_bn_feature_net_2D(self, include_top, padding, padding_mode, shape,
                               dilated, multires, location, data_format):
        n_features = 3
        n_dense_filters = 200

        # BAD: dilated=True, include_top=False
        # BAD: inputs != None

        with self.cached_session():
            K.set_image_data_format(data_format)
            model = featurenet.bn_feature_net_2D(
                include_top=include_top,
                dilated=dilated,
                input_shape=shape,
                n_features=n_features,
                n_dense_filters=n_dense_filters,
                padding=padding,
                padding_mode=padding_mode,
                multires=multires,
                VGG_mode=multires,
                location=location)
            self.assertEqual(len(model.output_shape), 4)
            output = n_features if include_top else n_dense_filters
            axis = 1 if data_format == 'channels_first' else -1
            self.assertEqual(model.output_shape[axis], output)

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            data_format=['channels_first', 'channels_last']))
    def test_bn_feature_net_2D_skip(self, data_format):
        receptive_field = 61
        n_features = 3
        n_dense_filters = 300
        input_shape = (256, 256, 1)
        n_skips = 1

        with self.cached_session():
            K.set_image_data_format(data_format)
            axis = 1 if data_format == 'channels_first' else -1

            fgbg_model = featurenet.bn_feature_net_skip_2D(
                receptive_field=receptive_field,
                input_shape=input_shape,
                n_features=n_features,
                n_dense_filters=n_dense_filters,
                n_skips=n_skips,
                last_only=False)

            self.assertIsInstance(fgbg_model.output, list)
            self.assertEqual(len(fgbg_model.output), n_skips + 1)

            model = featurenet.bn_feature_net_skip_2D(
                receptive_field=receptive_field,
                input_shape=input_shape,
                fgbg_model=fgbg_model,
                n_features=n_features,
                n_dense_filters=n_dense_filters,
                n_skips=n_skips,
                last_only=True)

            self.assertEqual(len(model.output_shape), 4)
            self.assertEqual(model.output_shape[axis], n_features)

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters([
        {
            'testcase_name': 'dilated_include_top',
            'include_top': True,
            'dilated': True,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 33, 33, 1),
            'data_format': 'channels_last',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_include_top_cf',
            'include_top': True,
            'dilated': True,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 33, 33, 1),
            'data_format': 'channels_first',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        # {
        #     'testcase_name': 'not_dilated_include_top',
        #     'include_top': True,
        #     'dilated': False,
        #     'padding_mode': 'reflect',
        #     'padding': True,
        #     'multires': False,
        #     'location': False,
        #     'data_format': 'channels_last',
        # },
        # {
        #     'testcase_name': 'not_dilated_include_top_cf',
        #     'include_top': True,
        #     'dilated': False,
        #     'padding_mode': 'reflect',
        #     'padding': True,
        #     'multires': False,
        #     'location': False,
        #     'data_format': 'channels_first',
        # },
        {
            'testcase_name': 'dilated_no_top',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_cf',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_first',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'not_dilated_no_top',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 33, 33, 1),
            'data_format': 'channels_last',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'not_dilated_no_top_cf',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 33, 33, 1),
            'data_format': 'channels_first',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'not_dilated_no_top_location',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': True,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'not_dilated_no_top_location_cf',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': True,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_first',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'not_dilated_no_top_multires',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': False,
            'shape': (10, 33, 33, 1),
            'data_format': 'channels_last',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'not_dilated_no_top_multires_cf',
            'include_top': False,
            'dilated': False,
            'padding_mode': 'reflect',
            'padding': False,
            'multires': True,
            'location': False,
            'shape': (10, 33, 33, 1),
            'data_format': 'channels_first',
            'temporal': None,
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_conv3D_nonresidual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'conv',
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_conv3D_residual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'conv',
            'residual': True,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_convGRU2D_nonresidual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'gru',
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_convgru2D_nonresidual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'GRU',
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_convGRU2D_residual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'gru',
            'residual': True,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_convlstm2D_nonresidual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'lstm',
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_convLSTM2D_nonresidual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'LSTM',
            'residual': False,
            'temporal_kernel_size': 3,
        },
        {
            'testcase_name': 'dilated_no_top_convLSTM2D_residual',
            'include_top': False,
            'dilated': True,
            'padding_mode': 'zero',
            'padding': False,
            'multires': False,
            'location': False,
            'shape': (10, 32, 32, 1),
            'data_format': 'channels_last',
            'temporal': 'lstm',
            'residual': True,
            'temporal_kernel_size': 3,
        },
    ])
    def test_bn_feature_net_3D(self, include_top, padding, padding_mode, shape,
                               dilated, multires, location, data_format,
                               temporal, residual, temporal_kernel_size):
        n_features = 3
        n_dense_filters = 200
        n_frames = 5
        # input_shape = (10, 32, 32, 1)

        with self.cached_session():
            K.set_image_data_format(data_format)
            model = featurenet.bn_feature_net_3D(
                include_top=include_top,
                dilated=dilated,
                n_frames=n_frames,
                input_shape=shape,
                n_features=n_features,
                n_dense_filters=n_dense_filters,
                padding=padding,
                padding_mode=padding_mode,
                multires=multires,
                VGG_mode=multires,
                location=location,
                temporal=temporal,
                residual=residual,
                temporal_kernel_size=temporal_kernel_size)
            self.assertEqual(len(model.output_shape), 5 if dilated else 2)
            channel_axis = 1 if data_format == 'channels_first' else -1
            self.assertEqual(model.output_shape[channel_axis], n_features)

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            data_format=['channels_first', 'channels_last']))
    def test_bn_feature_net_3D_skip(self, data_format):
        receptive_field = 61
        n_features = 3
        n_dense_filters = 300
        input_shape = (10, 32, 32, 1)
        n_skips = 1
        temporal = None
        residual = False
        temporal_kernel_size = 3

        with self.cached_session():
            K.set_image_data_format(data_format)
            axis = 1 if data_format == 'channels_first' else -1

            fgbg_model = featurenet.bn_feature_net_skip_3D(
                receptive_field=receptive_field,
                input_shape=input_shape,
                n_features=n_features,
                n_dense_filters=n_dense_filters,
                n_skips=n_skips,
                last_only=False,
                temporal=temporal,
                residual=residual,
                temporal_kernel_size=temporal_kernel_size)

            self.assertIsInstance(fgbg_model.output, list)
            self.assertEqual(len(fgbg_model.output), n_skips + 1)

            model = featurenet.bn_feature_net_skip_3D(
                receptive_field=receptive_field,
                input_shape=input_shape,
                fgbg_model=fgbg_model,
                n_features=n_features,
                n_dense_filters=n_dense_filters,
                n_skips=n_skips,
                last_only=True,
                temporal=temporal,
                residual=residual,
                temporal_kernel_size=temporal_kernel_size)

            self.assertEqual(len(model.output_shape), 5)
            self.assertEqual(model.output_shape[axis], n_features)
