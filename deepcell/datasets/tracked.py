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
"""Timelapse datasets of a nuclear label including the raw images and
ground truth segmentation masks annotated to track cell lineages"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras.utils.data_utils import get_file

from deepcell.utils.tracking_utils import load_trks
from deepcell.datasets import Dataset


def load_data():
    """Loads tracked nuclei movies for building tracking models.

    The dataset consists of 512x512 nuclear fluorescent movies,
    along with instance masks and a JSON file containing cell
    lineage data.

    Returns:
        dict: ``{'X': np.array, 'y': np.array, 'lineages': dict}``.
    """
    dirname = os.path.join('datasets', 'tracked-nuclear')
    base = 'https://deepcell-data.s3.amazonaws.com/tracked/nuclei/'
    files = ['train.trks', 'val.trks']

    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))

    train_data = load_trks(paths[0])
    test_data = load_trks(paths[1])

    return train_data, test_data


# The following datasets were used to benchmark DeepCell-Tracking in "Accurate
# cell tracking and lineage construction in live-cell imaging experiments with
# deep learning" (2019)

# TODO: Correct all 4 full datasets (above) to improve training accuracy and
#        allow for dynamic train/val/test split with seed values

# pylint: disable=line-too-long
#:
nih_3t3_bench = Dataset(
    path='3T3_NIH_benchmarks.trks',
    url='https://deepcell-data.s3.amazonaws.com/tracking_manuscript_benchmarking/GT_tracks/3T3_NIH_benchmarks.trks',
    file_hash='fb4a6afc3fc10db0d9b07dd8db516eaf',
    metadata={}
)

#:
hek293_bench = Dataset(
    path='HEK293_generic_benchmarks.trks',
    url='https://deepcell-data.s3.amazonaws.com/tracking_manuscript_benchmarking/GT_tracks/HEK293_generic_benchmarks.trks',
    file_hash='b895098641c05655d79af9437962184f',
    metadata={}
)

#:
hela_s3_bench = Dataset(
    path='HeLa_S3_benchmarks.trks',
    url='https://deepcell-data.s3.amazonaws.com/tracking_manuscript_benchmarking/GT_tracks/HeLa_S3_benchmarks.trks',
    file_hash='ab3bc9f1a1bd0b0f93bbd20690d51585',
    metadata={}
)

#:
raw2647_bench = Dataset(
    path='RAW2647_generic_benchmarks.trks',
    url='https://deepcell-data.s3.amazonaws.com/tracking_manuscript_benchmarking/GT_tracks/RAW2647_generic_benchmarks.trks',
    file_hash='d832a462c1d476c7f8b9c78891ab3881',
    metadata={}
)
