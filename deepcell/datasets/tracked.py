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
        path = 'hek293.trks'
        url = 'https://deepcell-data.s3.amazonaws.com/tracked/HEK293.trks'
        file_hash = 'd5c563ab5866403836f2dcbe249c640f'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class hela_s3(Dataset):

    def __init__(self):
        path = 'HeLa_S3.trks'
        url = 'https://deepcell-data.s3.amazonaws.com/tracked/HeLa_S3.trks'
        file_hash = '590ee37d3c703cfe029a2e60c9dc777b'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class nih_3t3(Dataset):

    def __init__(self):
        path = '3T3_NIH.trks'
        url = 'https://deepcell-data.s3.amazonaws.com/tracked/3T3_NIH.trks'
        file_hash = '0d90ad370e1cb9655727065ada3ded65'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)
