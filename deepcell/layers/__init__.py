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
"""Custom Layers"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepcell.layers import convolutional_recurrent
from deepcell.layers import location
from deepcell.layers import normalization
from deepcell.layers import pooling
from deepcell.layers import tensor_product
from deepcell.layers import padding
from deepcell.layers import upsample

from deepcell.layers.convolutional_recurrent import ConvGRU2D
from deepcell.layers.location import Location2D
from deepcell.layers.location import Location3D
from deepcell.layers.normalization import ImageNormalization2D
from deepcell.layers.normalization import ImageNormalization3D
from deepcell.layers.pooling import DilatedMaxPool2D
from deepcell.layers.pooling import DilatedMaxPool3D
from deepcell.layers.temporal import Comparison, DeltaReshape, Unmerge
from deepcell.layers.temporal import TemporalMerge
from deepcell.layers.tensor_product import TensorProduct
from deepcell.layers.padding import ReflectionPadding2D
from deepcell.layers.padding import ReflectionPadding3D
from deepcell.layers.upsample import UpsampleLike

del absolute_import
del division
del print_function
