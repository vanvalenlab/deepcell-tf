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


def split_dataset(dataset, validation_data_fraction):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation
    dataset using given ratio. Fractions are rounded up to two decimal places.

    Args:
        dataset (tf.data.Dataset): the input dataset to split.
        validation_data_fraction (float): the fraction of the validation data
            between 0 and 1.
    
    Returns:
        (tf.data.Dataset, tf.data.Dataset): a tuple of (training, validation).
    """
    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError('validation_data_fraction must be ∈ [0,1].')

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)
    return train_dataset, validation_dataset


from deepcell.data import tracking


del absolute_import
del division
del print_function
