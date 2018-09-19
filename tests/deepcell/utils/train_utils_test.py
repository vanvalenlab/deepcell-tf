# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Tests for train_utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import keras
from tensorflow.python.platform import test

from deepcell.utils.train_utils import rate_scheduler


class TrainUtilsTest(test.TestCase):
    def test_rate_scheduler(self):
        # if decay is small, learning rate should decrease as epochs increase
        rs = rate_scheduler(lr=.001, decay=.95)
        assert rs(1) > rs(2)
        # if decay is large, learning rate should increase as epochs increase
        rs = rate_scheduler(lr=.001, decay=1.05)
        assert rs(1) < rs(2)
        # if decay is 1, learning rate should not change
        rs = rate_scheduler(lr=.001, decay=1)
        assert rs(1) == rs(2)

if __name__ == '__main__':
    test.main()
