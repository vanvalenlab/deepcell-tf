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

import timeit

import tensorflow as tf
from tensorflow.keras import backend as K


class InferenceTimer(tf.keras.callbacks.Callback):
    """Callback to log inference speed per epoch."""

    def __init__(self):
        super(InferenceTimer, self).__init__()
        self._batch_times = []
        self._timer = None
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_epoch_begin(self, epoch, logs=None):
        self._batch_times = []

    def on_train_batch_begin(self, batch, logs=None):
        self._timer = timeit.default_timer()

    def on_train_batch_end(self, batch, logs=None):
        t = timeit.default_timer() - self._timer
        self._batch_times.append(t)

    def on_epoch_end(self, epoch, logs=None):
        avg = np.mean(self._batch_times)
        std = np.std(self._batch_times)
        print('Finished epoch {} with an average speed of {}s ± {}s.'.format(
            epoch, avg, std))
