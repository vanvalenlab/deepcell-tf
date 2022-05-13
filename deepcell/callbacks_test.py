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
"""Custom Callbacks for DeepCell"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.framework import test_util as tf_test_util

from deepcell import callbacks


class TestInferenceTimer(keras_parameterized.TestCase):
    """Callback to log inference speed per epoch."""

    @keras_parameterized.run_all_keras_modes
    def test_inference_time_logging(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1))
        model.compile(
            'sgd',
            loss='mse',
            run_eagerly=testing_utils.should_run_eagerly())

        x = tf.ones((200, 3))
        y = tf.zeros((200, 2))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)
        expected_log = r'(.*Average inference.*)+'

        cbks = [callbacks.InferenceTimer()]

        with self.captureWritesToStream(sys.stdout) as printed:
            y = model.call(x)
            model.fit(dataset, epochs=2, steps_per_epoch=10, callbacks=cbks)
            self.assertRegex(printed.contents(), expected_log)
