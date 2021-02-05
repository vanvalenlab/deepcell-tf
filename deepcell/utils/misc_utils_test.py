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
"""Tests for misc_utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

from deepcell.utils import misc_utils


class MiscUtilsTest(test.TestCase):
    def test_sorted_nicely(self):
        # test image file sorting
        expected = ['test_001_dapi', 'test_002_dapi', 'test_003_dapi']
        unsorted = ['test_003_dapi', 'test_001_dapi', 'test_002_dapi']
        self.assertListEqual(expected, misc_utils.sorted_nicely(unsorted))
        # test montage folder sorting
        expected = ['test_0_0', 'test_1_0', 'test_1_1']
        unsorted = ['test_1_1', 'test_0_0', 'test_1_0']
        self.assertListEqual(expected, misc_utils.sorted_nicely(unsorted))

    def test_get_sorted_keys(self):
        d = {'C1': 1, 'C3': 2, 'C2': 3}
        self.assertListEqual(misc_utils.get_sorted_keys(d), ['C1', 'C2', 'C3'])


if __name__ == '__main__':
    test.main()
