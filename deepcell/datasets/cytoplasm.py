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
"""Instance masks of fluorescent cytoplasm images"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file


def load_data():
    """Loads cytoplasm image segmentation training data.

    The dataset consists of 512x512 fluorescent cytoplasm images,
    with along with a instance label mask.

    Returns:
        tuple(numpy.array): ``(x_train, y_train), (x_test, y_test)``.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = nuclear.load_data()
    assert x_train.shape[1:] == (512, 512, 1)
    assert x_test.shape[1:] == (512, 512, 1)
    assert y_train.shape[1:] == (512, 512, 1)
    assert y_test.shape[1:] == (512, 512, 1)
    ```

    """
    dirname = os.path.join('datasets', 'cytoplasm-fluorescence')
    base = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/'
    files = ['train.npz', '_val.npz']

    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))

    data = np.load(paths[0])
    x_train, y_train = data['X'], data['y']

    data = np.load(paths[1])
    x_test, y_test = data['X'], data['y']

    return (x_train, y_train), (x_test, y_test)
