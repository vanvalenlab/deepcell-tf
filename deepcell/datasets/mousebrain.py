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
"""Mouse Brain Nuclear 3D Dataset from the Long Cai Group"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

try:
    from tensorflow.keras.utils.data_utils import get_file
except ImportError:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file

from deepcell.utils.data_utils import get_data


def load_data(path='mousebrain.npz', test_size=.2, seed=0):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    basepath = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
    prefix_path = os.path.join(*path.split(os.path.sep)[:-1])
    data_dir = os.path.join(basepath, prefix_path)
    if not os.path.exists(data_dir):
        if not os.path.isdir(data_dir):
            raise IOError('{} exists but is not a directory'.format(data_dir))
        os.makedirs(data_dir)

    path = get_file(path,
                    origin='https://deepcell-data.s3.amazonaws.com/nuclei/mousebrain.npz',
                    file_hash='9c91304f7da7cc5559f46b2c5fc2eace')

    train_dict, test_dict = get_data(path, test_size=test_size, seed=seed)

    x_train, y_train = train_dict['X'], train_dict['y']
    x_test, y_test = test_dict['X'], test_dict['y']
    return (x_train, y_train), (x_test, y_test)
