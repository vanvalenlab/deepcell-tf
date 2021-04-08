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

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K


class InferenceTimer(tf.keras.callbacks.Callback):
    """Callback to log inference speed per epoch."""

    def __init__(self, samples=100):
        super(InferenceTimer, self).__init__()
        self._samples = int(samples)
        self._batch_times = []
        self._samples_seen = []
        self._timer = None

    def on_predict_begin(self, epoch, logs=None):
        self._batch_times = []
        self._samples_seen = []

    def on_predict_batch_begin(self, batch, logs=None):
        self._timer = timeit.default_timer()

    def on_predict_batch_end(self, batch, logs=None):
        t = timeit.default_timer() - self._timer
        self._batch_times.append(t)
        outputs = logs.get('outputs', np.empty((1,)))
        if isinstance(self.model.output_shape, list):
            outputs = outputs[0]
        self._samples_seen.append(outputs.shape[0])

    def on_predict_end(self, logs=None):
        total_samples = np.sum(self._samples_seen)

        per_sample = [t / float(s) for t, s in
                      zip(self._batch_times, self._samples_seen)]

        avg = np.mean(per_sample)
        std = np.std(per_sample)

        print('Average inference speed per sample for %s total samples: '
              '%0.5fs ± %0.5fs.' % (total_samples, avg, std))

    def on_epoch_end(self, epoch, logs=None):
        shape = tuple([self._samples] + list(self.model.input_shape[1:]))
        test_batch = np.random.random(shape)
        self.model.predict(test_batch, callbacks=self)
