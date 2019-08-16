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


class nih_3t3(Dataset):

    def __init__(self):
        path = 'nih_3t3-cytoplasm.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_3T3_s0_fluorescent_cyto_medium_stitched_2D_512.npz'
        file_hash = '6d3278cff6a82178dc40984e86f71ffd39c465b196e3d0b4b933949cc260adc9'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class A549(Dataset):

    def __init__(self):
        path = 'A549-cytoplasm.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_A549_s0_fluorescent_cyto_medium_stitched_2D.npz'
        file_hash = '33741ff643b1c8c017269663978ab8d52f833bfb65156fb66defa325e5316e74'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class CHO(Dataset):

    def __init__(self):
        path = 'CHO-cytoplasm.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_CHO_s0_fluorescent_cyto_medium_stitched_2D_512.npz'
        file_hash = 'd56029039a94c3c5ffaf926796108b87d2f12792b30010af52136ef4281dbbff'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class hela_s3(Dataset):

    def __init__(self):
        path = 'hela_s3-cytoplasm.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_HeLa-S3_s0_fluorescent_cyto_medium_stitched_2D_512.npz'
        file_hash = '1ed6c47db02687e64a34d305a30677ad0a286106227c6d8992e23ca27ce6e098'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class hela(Dataset):

    def __init__(self):
        path = 'hela-cytoplasm.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_HeLa_s0_fluorescent_cyto_medium_stitched_2D_512.npz'
        file_hash = '8dc76a2d4a5f384727e31daa9f25b6d861e64bd1775aec7ee42bea2cdf2b0527'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class pc3(Dataset):

    def __init__(self):
        path = 'pc3-cytoplasm.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_PC3_s0_fluorescent_cyto_medium_stitched_2D_512.npz'
        file_hash = '194016feb25e97bdf0b1e335acab3c217d14ae2679a7ac7d3f6204ce4a864560'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)
