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
"""Package for single cell image segmentation with convolutional neural networks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepcell._version import __version__

from deepcell import applications
from deepcell import callbacks
from deepcell import datasets
from deepcell import layers
from deepcell import losses
from deepcell import initializers
from deepcell import image_generators
from deepcell import model_zoo
from deepcell import running
from deepcell import tracking
from deepcell import training
from deepcell import utils
from deepcell import metrics

from deepcell.layers import *
from deepcell.image_generators import *
from deepcell.model_zoo import *
from deepcell.running import get_cropped_input_shape
from deepcell.running import process_whole_image
from deepcell.training import train_model_conv
from deepcell.training import train_model_sample
from deepcell.training import train_model_siamese_daughter
from deepcell.training import train_model_retinanet
from deepcell.utils import *

del absolute_import
del division
del print_function
