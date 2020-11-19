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
"""Builtin Datasets"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.keras.utils import get_file

from deepcell.utils.data_utils import get_data


class Dataset(object):  # pylint: disable=useless-object-inheritance

    """General class for downloading datasets from S3.

    Args:
        path (str): path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        url (str): URL of dataset in S3.
        file_hash (str): md5hash for checking validity of cached file.
        metadata (dict): miscellaneous other data for dataset
    """

    def __init__(self,
                 path,
                 url,
                 file_hash,
                 metadata):
        self.path = path
        self.url = url
        self.file_hash = file_hash
        self.metadata = metadata

    def _load_data(self, path, mode, test_size=0.2, seed=0):
        """Loads dataset.

        Args:
            test_size (float): fraction of data to reserve as test data
            seed (int): the seed for randomly shuffling the dataset

        Returns:
            tuple: (x_train, y_train), (x_test, y_test).
        """
        basepath = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
        prefix = path.split(os.path.sep)[:-1]
        data_dir = os.path.join(basepath, *prefix) if prefix else basepath
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        elif not os.path.isdir(data_dir):
            raise IOError('{} exists but is not a directory'.format(data_dir))

        path = get_file(path,
                        origin=self.url,
                        file_hash=self.file_hash)

        train_dict, test_dict = get_data(
            path,
            mode=mode,
            test_size=test_size,
            seed=seed)

        x_train, y_train = train_dict['X'], train_dict['y']
        x_test, y_test = test_dict['X'], test_dict['y']
        return (x_train, y_train), (x_test, y_test)

    def load_data(self, path=None, test_size=0.2, seed=0):
        """Loads dataset.

        Args:
            path (str): filepath to save the data locally.
            test_size (float): fraction of data to reserve as test data
            seed (int): the seed for randomly shuffling the dataset

        Returns:
            tuple: (x_train, y_train), (x_test, y_test).
        """
        path = path if path else self.path
        return self._load_data(path, 'sample', test_size=test_size, seed=seed)

    def load_tracked_data(self, path=None, test_size=0.2, seed=0):
        """Loads dataset using "siamese_daughters" mode.

        Args:
            path (str): filepath to save the data locally.
            test_size (float): fraction of data to reserve as test data
            seed (int): the seed for randomly shuffling the dataset

        Returns:
            tuple: (x_train, y_train), (x_test, y_test).
        """
        path = path if path else self.path
        return self._load_data(path, 'siamese_daughters', test_size=test_size, seed=seed)

# pylint: disable=line-too-long


#:
hek293 = Dataset(
    path='HEK293.npz',
    url='https://deepcell-data.s3-us-west-1.amazonaws.com/nuclei/HEK293.npz',
    file_hash='6221fa459350cd1e45ce6c9145264329',
    metadata={}
)

#:
nih_3t3 = Dataset(
    path='3T3_NIH.npz',
    url='https://deepcell-data.s3.amazonaws.com/nuclei/3T3_NIH.npz',
    file_hash='f6520df218847fa56be2de0d3552c8a2',
    metadata={}
)

#:
hela_s3 = Dataset(
    path='HeLa_S3.npz',
    url='https://deepcell-data.s3.amazonaws.com/nuclei/HeLa_S3.npz',
    file_hash='759d28d87936fd59b250dea3b126b647',
    metadata={}
)

#:
mibi = Dataset(
    path='mibi_original.npz',
    url='https://deepcell-data.s3.amazonaws.com/mibi/mibi_original.npz',
    file_hash='8b09a6bb143deb1912ada65742dfc847',
    metadata={}
)

#:
mousebrain = Dataset(
    path='mousebrain.npz',
    url='https://deepcell-data.s3.amazonaws.com/nuclei/mousebrain.npz',
    file_hash='9c91304f7da7cc5559f46b2c5fc2eace',
    metadata={}
)

#:
multiplex_tissue = Dataset(
    path='20200810_tissue_dataset.npz',
    url='https://deepcell-data.s3.amazonaws.com/multiplex/20200810_tissue_dataset.npz',
    file_hash='1e573b72123fd86e45433402094bf0d0',
    metadata={}
)

# pylint: disable=wrong-import-position
from deepcell.datasets import cytoplasm
from deepcell.datasets import phase
from deepcell.datasets import tracked
# pylint: enable=wrong-import-position

del absolute_import
del division
del print_function
