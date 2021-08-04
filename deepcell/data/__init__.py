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
"""Data utilities using ``tf.data``."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def split_dataset(dataset, val_size, test_size=0):
    """
    Splits a dataset of type tf.data.Dataset into a training, validation, and
    optionally test dataset using given ratios. Fractions are rounded up to
    two decimal places.

    Inspired by: https://stackoverflow.com/a/59696126

    Args:
        dataset (tf.data.Dataset): the input dataset to split.
        val_size (float): the fraction of the validation data
            between 0 and 1.
        test_size (float): the fraction of the test data between 0
            and 1.

    Returns:
        (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset): a tuple of
            (training, validation, test).
    """
    val_percent = round(val_size * 100)
    if not 0 <= val_percent <= 100:
        raise ValueError('val_size must be ∈ [0,1].')

    test_percent = round(test_size * 100)
    if not 0 <= test_percent <= 100:
        raise ValueError('test_size must be ∈ [0,1].')

    if val_percent + test_percent >= 100:
        raise ValueError('sum of val_size and '
                         'test_size must be ∈ [0,1].')

    dataset = dataset.enumerate()
    # TODO: Will cause issues if there are fewer than 100 records
    val_dataset = dataset.filter(lambda f, data: f % 100 <= val_percent)
    train_dataset = dataset.filter(lambda f, data:
                                   f % 100 > test_percent + val_percent)
    test_dataset = dataset.filter(lambda f, data: f % 100 > val_percent and
                                  f % 100 <= val_percent + test_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    val_dataset = val_dataset.map(lambda f, data: data)
    test_dataset = test_dataset.map(lambda f, data: data)
    return train_dataset, val_dataset, test_dataset


from deepcell.data import tracking


del absolute_import
del division
del print_function
