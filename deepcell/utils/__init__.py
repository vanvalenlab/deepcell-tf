# Copyright 2016-2023 The Van Valen Lab at the California Institute of
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


# compute_overlap has been moved to deepcell_toolbox
# leaving here for backwards compatibility
from deepcell_toolbox import compute_overlap

from deepcell.utils import backbone_utils
from deepcell.utils import data_utils
from deepcell.utils import io_utils
from deepcell.utils import misc_utils
from deepcell.utils import plot_utils
from deepcell.utils import tracking_utils
from deepcell.utils import train_utils
from deepcell.utils import transform_utils
from deepcell.utils import tfrecord_utils

# Globally-importable utils.
from deepcell.utils.data_utils import get_data
from ._auth import fetch_data
from deepcell.utils.misc_utils import sorted_nicely
from deepcell.utils.train_utils import rate_scheduler
from deepcell.utils.transform_utils import outer_distance_transform_2d
from deepcell.utils.transform_utils import outer_distance_transform_3d
from deepcell.utils.transform_utils import outer_distance_transform_movie
from deepcell.utils.transform_utils import inner_distance_transform_2d
from deepcell.utils.transform_utils import inner_distance_transform_3d
from deepcell.utils.transform_utils import inner_distance_transform_movie
from deepcell.utils.transform_utils import pixelwise_transform


def __getattr__(name):
    if name in {"export_model", "export_utils"}:
        import warnings
        import importlib

        warnings.warn(
            f"\n\n{name} is deprecated, use tf.keras.models.save_model directly.\n",
            DeprecationWarning,
            stacklevel=2,
        )

        export_utils = importlib.import_module("deepcell.utils.export_utils")
        if name == "export_model":
            return getattr(export_utils, name)
        if name == "export_utils":
            return export_utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
