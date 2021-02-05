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
"""Fluorescent cytoplasm datasets including
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

#:
all_cells = Dataset(
    path='all_fluorescent_cyto_fixed.npz',
    url='https://deepcell-data.s3-us-west-1.amazonaws.com/cytoplasm/fluorescent/20190903_all_fluorescent_cyto_512_contrast_adjusted_curated.npz',
    file_hash='6548b5b54cd940fb7ced864b4422eb96',
    metadata={'methods': methods}
)

#:
nih_3t3 = Dataset(
    path='nih_3t3-cytoplasm.npz',
    url='https://deepcell-data.s3-us-west-1.amazonaws.com/cytoplasm/fluorescent/nih_3t3-cytoplasm_fixed.npz',
    file_hash='a4e671a2c08e102f158903b288e88fff',
    metadata={'methods': methods}
)

#:
a549 = Dataset(
    path='A549-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/A549-cytoplasm_fixed.npz',
    file_hash='b5934b2424d2c0e91aedb577883bb0e5',
    metadata={'methods': methods}
)

#:
cho = Dataset(
    path='CHO-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/CHO-cytoplasm_fixed.npz',
    file_hash='949c1259d1383feb088f4cecdbc9a655',
    metadata={'methods': methods}
)

#:
hela_s3 = Dataset(
    path='hela_s3-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/hela_s3-cytoplasm_fixed.npz',
    file_hash='a2571357f5c2b238ed2ca9ec1329d8ea',
    metadata={'methods': methods}
)

#:
hela = Dataset(
    path='hela-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/hela-cytoplasm_fixed.npz',
    file_hash='40f301ffb315dab17491c06afbfdf641',
    metadata={'methods': methods}
)

#:
pc3 = Dataset(
    path='pc3-cytoplasm.npz',
    url='https://deepcell-data.s3.amazonaws.com/cytoplasm/fluorescent/pc3-cytoplasm_fixed.npz',
    file_hash='636a280367c52bc9b57eebed23661295',
    metadata={'methods': methods}
)
