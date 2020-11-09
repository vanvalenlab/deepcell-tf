# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
"""Tests for custom callbacks"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils

from deepcell.callbacks import RedirectModel


TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 1
NUM_HIDDEN = 5
BATCH_SIZE = 5


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class CallbacksTest(keras_parameterized.TestCase):

    def test_RedirectModel(self):
        with self.cached_session():
            np.random.seed(123)
            (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
                train_samples=TRAIN_SAMPLES,
                test_samples=TEST_SAMPLES,
                input_shape=(INPUT_DIM,),
                num_classes=NUM_CLASSES)

            y_test = keras.utils.np_utils.to_categorical(y_test)
            y_train = keras.utils.np_utils.to_categorical(y_train)
            model = keras.models.Sequential()
            model.add(
                keras.layers.Dense(
                    NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
            model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

            prediction_model = copy.copy(model)

            cbks = [
                RedirectModel(
                    keras.callbacks.Callback(),
                    prediction_model
                )
            ]
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=5,
                verbose=0)
