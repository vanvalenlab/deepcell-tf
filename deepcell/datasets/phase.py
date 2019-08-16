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
        'humidified 5% CO2 atmosphere.  When 70-80% confluent, cells were '
        'passaged and seeded onto fibronectin coated glass bottom 96-well '
        'plates at 10,000-20,000 cells/well. The seeded cells were then '
        'incubated for 1-2 hours to allow for cell adhesion to the bottom of '
        'the well plate before imaging.',
    'Imaging':
        'Cells were imaged on a Nikon Eclipse Ti-2 fluorescence microscope at '
        '20x and 40x for NIH3t3 and Raw293.6 cells respectively. The well '
        'plate was placed in a Nikon incubated stage with an Oko labs '
        'environment controller set to 37C and 5% CO2. Each data set was '
        'generated using the Nikon jobs function to collect a z-stack of phase '
        'images.'
}

nih_3t3 = Dataset(
    path='nih_3t3-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_3T3_s0_phase_medium_stitched_2D_512.np',
    file_hash='b0dc7fa28d6ec4dec25150187b9629330689372da40f730042f8e0824df4da2e',
    metadata={'methods': methods}
)


A549 = Dataset(
    path='A549-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_A549_s0_phase_medium_stitched_2D_512.npz',
    file_hash='4e2a17ed2083ffa7e9b64824c27591bf776257bae2b07639226aa2b9900fbb33',
    metadata={'methods': methods}
)


CHO = Dataset(
    path='CHO-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_CHO_s0_phase_medium_stitched_2D_512.npz',
    file_hash='39aad99486825a856da14d87b48ac9b00dd176b7e54c9fc928489ee464828bcd',
    metadata={'methods': methods}
)


HeLa_S3 = Dataset(
    path='HeLa_S3-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_HeLa-S3_s0_phase_medium_stitched_2D_512.npz',
    file_hash='505c42d89005186d0eb4d76740bb829c13220b6a96b0bfdc3980eaf95a359293',
    metadata={'methods': methods}
)


HeLa = Dataset(
    path='HeLa-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_HeLa_s0_phase_medium_stitched_2D_512.npz',
    file_hash='e4e92e2611cd4bf087d8489db6fed35c893566f0d3fe859e366511f087a3f64c',
    metadata={'methods': methods}
)


PC3 = Dataset(
    path='PC3-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/AM_PC3_s0_phase_medium_stitched_2D_512.npz',
    file_hash='3d2d5106e1d1437e4874c6c49c773dbacc2437d5556212f7c2d532a820e5a03e',
    metadata={'methods': methods}
)
