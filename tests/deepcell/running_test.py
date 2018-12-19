# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
# ============================================================================
"""Tests for running functions
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test

from deepcell import layers
from deepcell import model_zoo
from deepcell import running


class RunningTests(test.TestCase):

    def test_get_cropped_input_shape(self):
        # test 2D images
        img_w, img_h = 30, 30
        batches = 5
        n_crops = 4
        field = 11
        # test channels_last
        keras.backend.set_image_data_format('channels_last')
        X = np.random.random((batches, img_w, img_h, 1))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field)

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (new_w, new_h, 1))

        # test channels_first
        X = np.random.random((batches, 1, img_w, img_h))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field,
            data_format='channels_first')

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (1, new_w, new_h))

        # test 3D images
        frames = 30

        # test channels_last
        X = np.random.random((batches, frames, img_w, img_h, 1))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field,
            data_format='channels_last')

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (frames, new_w, new_h, 1))

        # test channels_first
        X = np.random.random((batches, 1, frames, img_w, img_h))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field,
            data_format='channels_first')

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (1, frames, new_w, new_h))

    def test_get_padding_layers(self):
        # test reflection padding layer
        model = keras.models.Sequential()
        model.add(layers.ReflectionPadding2D(padding=(30, 30)))
        model.add(keras.layers.Dense(4))

        padded = running.get_padding_layers(model)
        self.assertEqual(len(padded), 1)

        # test zero padding layer
        model = keras.models.Sequential()
        model.add(keras.layers.ZeroPadding2D(padding=(30, 30)))
        model.add(keras.layers.Dense(4))

        padded = running.get_padding_layers(model)
        self.assertEqual(len(padded), 1)

        # test no padding layer
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(4))

        padded = running.get_padding_layers(model)
        self.assertEqual(len(padded), 0)


if __name__ == '__main__':
    test.main()
