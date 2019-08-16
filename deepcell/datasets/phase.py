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
        path = 'nih_3t3-phase.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_3T3_s0_phase_medium_stitched_2D_512.np'
        file_hash = 'b0dc7fa28d6ec4dec25150187b9629330689372da40f730042f8e0824df4da2e'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class A549(Dataset):

    def __init__(self):
        path = 'A549-phase.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_A549_s0_phase_medium_stitched_2D_512.npz'
        file_hash = '4e2a17ed2083ffa7e9b64824c27591bf776257bae2b07639226aa2b9900fbb33'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class CHO(Dataset):

    def __init__(self):
        path = 'CHO-phase.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_CHO_s0_phase_medium_stitched_2D_512.npz'
        file_hash = '39aad99486825a856da14d87b48ac9b00dd176b7e54c9fc928489ee464828bcd'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class HeLa_S3(Dataset):

    def __init__(self):
        path = 'HeLa_S3-phase.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_HeLa-S3_s0_phase_medium_stitched_2D_512.npz'
        file_hash = '505c42d89005186d0eb4d76740bb829c13220b6a96b0bfdc3980eaf95a359293'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class HeLa(Dataset):

    def __init__(self):
        path = 'HeLa-phase.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_HeLa_s0_phase_medium_stitched_2D_512.npz'
        file_hash = 'e4e92e2611cd4bf087d8489db6fed35c893566f0d3fe859e366511f087a3f64c'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)


class PC3(Dataset):

    def __init__(self):
        path = 'PC3-phase.npz'
        url = 'https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_PC3_s0_phase_medium_stitched_2D_512.npz'
        file_hash = '3d2d5106e1d1437e4874c6c49c773dbacc2437d5556212f7c2d532a820e5a03e'
        metadata = {}

        super(Dataset, self).__init__(
            path, url, file_hash, metadata)
