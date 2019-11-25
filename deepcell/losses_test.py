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
"""Tests for custom loss functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test

from deepcell import losses

try:
    import h5py  # pylint:disable=wrong-import-position
except ImportError:
    h5py = None


ALL_LOSSES = [
    losses.categorical_crossentropy,
    losses.weighted_categorical_crossentropy,
    losses.sample_categorical_crossentropy,
    losses.weighted_focal_loss,
    losses.smooth_l1,
    losses.focal,
    # losses.dice_coef_loss,
    # losses.discriminative_instance_loss
]


class _MSEMAELoss(object):
    """Loss function with internal state, for testing serialization code."""

    def __init__(self, mse_fraction):
        self.mse_fraction = mse_fraction

    def __call__(self, y_true, y_pred, sample_weight=None):
        return (self.mse_fraction * keras.losses.mean_squared_error(y_true, y_pred) +
                (1 - self.mse_fraction) * keras.losses.mean_absolute_error(y_true, y_pred))

    def get_config(self):
        return {'mse_fraction': self.mse_fraction}


class KerasLossesTest(test.TestCase):

    def test_objective_shapes_3d(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.random((5, 6, 7)))
            y_b = keras.backend.variable(np.random.random((5, 6, 7)))
            for obj in ALL_LOSSES:
                objective_output = obj(y_a, y_b)
                print(obj)
                self.assertListEqual(objective_output.get_shape().as_list(), [5, 6])

    def test_objective_shapes_2d(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.random((6, 7)))
            y_b = keras.backend.variable(np.random.random((6, 7)))
            for obj in ALL_LOSSES:
                objective_output = obj(y_a, y_b)
                print(obj)
                self.assertListEqual(objective_output.get_shape().as_list(), [6])

    def test_cce_one_hot(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.randint(0, 7, (5, 6)))
            y_b = keras.backend.variable(np.random.random((5, 6, 7)))
            objective_output = keras.losses.sparse_categorical_crossentropy(y_a, y_b)
            self.assertEqual(keras.backend.eval(objective_output).shape, (5, 6))

            y_a = keras.backend.variable(np.random.randint(0, 7, (6,)))
            y_b = keras.backend.variable(np.random.random((6, 7)))
            objective_output = keras.losses.sparse_categorical_crossentropy(y_a, y_b)
            self.assertEqual(keras.backend.eval(objective_output).shape, (6,))

    def test_serialization(self):
        fn = keras.losses.get('mse')
        config = keras.losses.serialize(fn)
        new_fn = keras.losses.deserialize(config)
        self.assertEqual(fn, new_fn)

    def test_categorical_hinge(self):
        y_pred = keras.backend.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
        y_true = keras.backend.variable(np.array([[0, 1, 0], [1, 0, 0]]))
        expected_loss = ((0.3 - 0.2 + 1) + (0.7 - 0.1 + 1)) / 2.0
        loss = keras.backend.eval(keras.losses.categorical_hinge(y_true, y_pred))
        self.assertAllClose(expected_loss, np.mean(loss))

    def test_serializing_loss_class(self):
        orig_loss_class = _MSEMAELoss(0.3)
        with keras.utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
            serialized = keras.losses.serialize(orig_loss_class)

        with keras.utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
            deserialized = keras.losses.deserialize(serialized)
        self.assertIsInstance(deserialized, _MSEMAELoss)
        self.assertEqual(deserialized.mse_fraction, 0.3)

    def test_serializing_model_with_loss_class(self):
        tmpdir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, tmpdir)
        model_filename = os.path.join(tmpdir, 'custom_loss.h5')

        with self.cached_session():
            with keras.utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
                loss = _MSEMAELoss(0.3)
                inputs = keras.layers.Input((2,))
                outputs = keras.layers.Dense(1, name='model_output')(inputs)
                model = keras.models.Model(inputs, outputs)
                model.compile(optimizer='sgd', loss={'model_output': loss})
                model.fit(np.random.rand(256, 2), np.random.rand(256, 1))

                if h5py is None:
                    return

                model.save(model_filename)

            with keras.utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
                loaded_model = keras.models.load_model(model_filename)
                loaded_model.predict(np.random.rand(128, 2))


if __name__ == '__main__':
    test.main()
