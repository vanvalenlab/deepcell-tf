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

from deepcell.datasets import Dataset


# pylint: disable=line-too-long
methods = {
    'Cell culture':
        'For our cytoplasmic data, NIH3t3 and Raw263.7 cell lines were '
        'cultured in DMEM media supplemented with 10% FBS and 1x '
        'penicillin-streptomycin antibiotic. Cells were incubated at 37C in a '
        'humidified 5% CO2 atmosphere. When 70-80% confluent, cells were '
        'passaged and seeded onto fibronectin coated glass bottom 96-well '
        'plates at 10,000-20,000 cells/well. The seeded cells were then '
        'incubated for 1-2 hours to allow for cell adhesion to the bottom of '
        'the well plate before imaging.',
    'Nuclear and Cytoplasmic Fluorescent Labeling':
        'To stain cell cytoplasm, each well was triturated 5-10 times and '
        'then washed with serum free DMEM to remove dead or nonadherent cells. '
        '100uL of Cell Tracker CMFDA diluted to 2uM with serum free DMEM was '
        'added to each well and then incubated for 15 minutes at 37C. '
        'Cell Tracker was the aspirated and cells were then resuspended in '
        '100uL of Hoescht 33342 diluted to 20uM using phenol-free DMEM '
        'supplemented with 10% FBS and 1x penicillin-streptomycin. The cells '
        'were then incubated at 37C for 5 minutes. The cells were then washed '
        '2 times with phenol-free complete media and then resuspended in '
        '200uL of phenol-free complete media.',
    'Imaging':
        'Cells were imaged on a Nikon Eclipse Ti-2 fluorescence microscope at '
        '20x and 40x for NIH3t3 and Raw293.6 cells respectively. The well '
        'plate was placed in a Nikon incubated stage with an Oko labs '
        'environment controller set to 37C and 5% CO2. '
        'Cytoplasm labeled with Cell Tracker CMFDA was visualized using the '
        'Nikon Sola LED light source and a Semrock GFP-4050B filter cube. '
        'Nuclei labeled with Hoescht 33342 were visualized with the same light '
        'source and a Semrock DAPI-3060 filter cube. Each data set was '
        'generated using the Nikon jobs function to image all fluorophores '
        'and phase as well as a z-stack of phase images.'
}

all_cells = Dataset(
    path='20190903_all_fluorescent_cyto_512.npz',
    url='https://deepcell-data.s3-us-west-1.amazonaws.com/cytoplasm/fluorescent/20190903_all_fluorescent_cyto_512.npz',
    file_hash='810f8180185dea6169f01470126fae4e38511645267fe92115d592ca11e1835e',
    metadata={'methods': methods}
)

nih_3t3 = Dataset(
    path='nih_3t3-cytoplasm.npz',
    url='https://deepcell-data.s3-us-west-1.amazonaws.com/cytoplasm/fluorescent/AM_3T3_s0_fluorescent_cyto_medium_stitched_2D_512.npz',
    file_hash='d04630d87835d27f11d80c123eac3d77684c57dadf783158c44084b11fac1fb3',
    metadata={'methods': methods}
)


A549 = Dataset(
    path='A549-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_A549_s0_fluorescent_cyto_medium_stitched_2D_512.npz',
    file_hash='33741ff643b1c8c017269663978ab8d52f833bfb65156fb66defa325e5316e74',
    metadata={'methods': methods}
)


CHO = Dataset(
    path='CHO-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_CHO_s0_fluorescent_cyto_medium_stitched_2D_512.npz',
    file_hash='d56029039a94c3c5ffaf926796108b87d2f12792b30010af52136ef4281dbbff',
    metadata={'methods': methods}
)


hela_s3 = Dataset(
    path='hela_s3-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_HeLa-S3_s0_fluorescent_cyto_medium_stitched_2D_512.npz',
    file_hash='1ed6c47db02687e64a34d305a30677ad0a286106227c6d8992e23ca27ce6e098',
    metadata={'methods': methods}
)


hela = Dataset(
    path='hela-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_HeLa_s0_fluorescent_cyto_medium_stitched_2D_512.npz',
    file_hash='8dc76a2d4a5f384727e31daa9f25b6d861e64bd1775aec7ee42bea2cdf2b0527',
    metadata={'methods': methods}
)


pc3 = Dataset(
    path='pc3-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/AM_PC3_s0_fluorescent_cyto_medium_stitched_2D_512.npz',
    file_hash='194016feb25e97bdf0b1e335acab3c217d14ae2679a7ac7d3f6204ce4a864560',
    metadata={'methods': methods}
)
