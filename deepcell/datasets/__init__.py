# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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

del absolute_import
del division
del print_function

from deepcell.utils.data_utils import Dataset


class hek293(Dataset):

    def __init__(self):
        path = 'HEK293.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/nuclei/3T3_NIH.npz'
        file_hash = 'f6520df218847fa56be2de0d3552c8a2'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class nih_3t3(Dataset):

    def __init__(self):
        path = '3T3_NIH.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/nuclei/3T3_NIH.npz'
        file_hash = 'f6520df218847fa56be2de0d3552c8a2'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class hela_s3(Dataset):

    def __init__(self):
        path = 'HeLa_S3.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/nuclei/HeLa_S3.npz'
        file_hash = '759d28d87936fd59b250dea3b126b647'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class mibi(Dataset):

    def __init__(self):
        path = 'mibi_original.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/mibi/mibi_original.npz'
        file_hash = '8b09a6bb143deb1912ada65742dfc847'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class mousebrain(Dataset):

    def __init__(self):
        path = 'mousebrain.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/nuclei/mousebrain.npz'
        file_hash = '9c91304f7da7cc5559f46b2c5fc2eace'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)
