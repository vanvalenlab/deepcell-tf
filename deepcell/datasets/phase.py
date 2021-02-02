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
"""Phase datasets including
the raw images and ground truth segmentation masks"""

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

#:
all_cells = Dataset(
    path='all_phase_fixed.npz',
    url='https://deepcell-data.s3-us-west-1.amazonaws.com/cytoplasm/brightfield/20190813_all_phase_512_contrast_adjusted_curated.npz',
    file_hash='11f5c0cc0899905ea889463be3cd4773',
    metadata={'methods': methods}
)

#:
nih_3t3 = Dataset(
    path='nih_3t3-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/nih_3t3-phase_fixed.npz',
    file_hash='d1a3b5a548300ef8389cee8021f53957',
    metadata={'methods': methods}
)

#:
a549 = Dataset(
    path='A549-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/A549-phase_fixed.npz',
    file_hash='d1820a7057079a774a9def8ae4634e74',
    metadata={'methods': methods}
)

#:
cho = Dataset(
    path='CHO-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/CHO-phase_fixed.npz',
    file_hash='0d059506a9500e155d0fbfee64c43e21',
    metadata={'methods': methods}
)

#:
hela_s3 = Dataset(
    path='HeLa_S3-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/HeLa_S3-phase_fixed.npz',
    file_hash='8ee318c32e41c9ff0fccf40bcd9d993d',
    metadata={'methods': methods}
)

#:
hela = Dataset(
    path='HeLa-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/HeLa-phase_fixed.npz',
    file_hash='f16c22201d63d1ab856f066811b3dcfa',
    metadata={'methods': methods}
)

#:
pc3 = Dataset(
    path='PC3-phase.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/brightfield/PC3-phase_fixed.npz',
    file_hash='1be17d9f6dbb009eed542f56f8282edd',
    metadata={'methods': methods}
)
