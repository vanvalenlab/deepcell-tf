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
"""Deepcell Model Zoo Module"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepcell.model_zoo.featurenet import bn_feature_net_2D
from deepcell.model_zoo.featurenet import bn_feature_net_skip_2D
from deepcell.model_zoo.featurenet import bn_feature_net_3D
from deepcell.model_zoo.featurenet import bn_feature_net_skip_3D

from deepcell.model_zoo.featurenet import siamese_model

from deepcell.model_zoo.retinanet import RetinaNet
from deepcell.model_zoo.retinanet import retinanet
from deepcell.model_zoo.retinanet import retinanet_bbox

from deepcell.model_zoo.retinamask import RetinaMask
from deepcell.model_zoo.retinamask import retinamask
from deepcell.model_zoo.retinamask import retinamask_bbox

from deepcell.model_zoo.panopticnet import PanopticNet

del absolute_import
del division
del print_function
