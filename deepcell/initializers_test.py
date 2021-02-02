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
"""Tests for custom initializers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.python.keras import keras_parameterized

from deepcell.initializers import PriorProbability


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class InitializersTest(keras_parameterized.TestCase):

    def _runner(self, init, shape, target_mean=None, target_std=None,
                target_max=None, target_min=None):
        variable = K.variable(init(shape))
        output = K.get_value(variable)
        # Test serialization (assumes deterministic behavior).
        config = init.get_config()
        reconstructed_init = init.__class__.from_config(config)
        variable = K.variable(reconstructed_init(shape))
        output_2 = K.get_value(variable)
        self.assertAllClose(output, output_2, atol=1e-4)

    def test_prior_probability(self):
        tensor_shape = (8, 12, 99)
        # TODO: use self.test_session() if tf version >= 1.11.0
        with self.cached_session():
            self._runner(PriorProbability(probability=0.01),
                         tensor_shape, target_mean=0., target_std=1)
