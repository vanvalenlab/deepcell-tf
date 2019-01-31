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
"""Deepcell Utilities Module"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from deepcell.utils import data_utils
from deepcell.utils import export_utils
from deepcell.utils import io_utils
from deepcell.utils import misc_utils
from deepcell.utils import plot_utils
from deepcell.utils import testing_utils
from deepcell.utils import train_utils
from deepcell.utils import transform_utils
from deepcell.utils import retinanet_anchor_utils

try:
    from deepcell.utils import compute_overlap
except ImportError:
    warnings.warn('To use `compute_overlap`, the C extensions must be built '
                  'using `python setup.py build_ext --inplace`')

# Globally-importable utils.
from deepcell.utils.data_utils import get_data
from deepcell.utils.data_utils import make_training_data
from deepcell.utils.export_utils import export_model
from deepcell.utils.io_utils import get_immediate_subdirs
from deepcell.utils.io_utils import get_image
from deepcell.utils.io_utils import nikon_getfiles
from deepcell.utils.io_utils import get_image_sizes
from deepcell.utils.io_utils import get_images_from_directory
from deepcell.utils.misc_utils import sorted_nicely
from deepcell.utils.train_utils import rate_scheduler
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.utils.transform_utils import distance_transform_3d
from deepcell.utils.transform_utils import deepcell_transform

del absolute_import
del division
del print_function
