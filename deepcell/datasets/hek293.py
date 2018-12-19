# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""HEK293 Nuclear Dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from tensorflow.python.keras.utils.data_utils import get_file
except ImportError:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file

from deepcell.utils.data_utils import get_data


def load_data(path='HEK293.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://deepcell-data.s3.amazonaws.com/nuclei/HEK293.npz',
                    file_hash='c0bbfba54b90e63a2010133a198e6e63')

    train_dict, test_dict = get_data(path, seed=0, test_size=.2)

    x_train, y_train = train_dict['X'], train_dict['y']
    x_test, y_test = test_dict['X'], test_dict['y']
    return (x_train, y_train), (x_test, y_test)
