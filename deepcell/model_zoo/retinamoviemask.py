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
"""MaskRCNN models adapted from https://github.com/fizyr/keras-maskrcnn"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Concatenate, Lambda, Dense, Layer
from tensorflow.python.keras.layers import BatchNormalization, Activation, Softmax
from tensorflow.python.keras.layers import TimeDistributed, Conv2D, MaxPool2D
from tensorflow.python.keras.models import Model
try:
    from tensorflow.python.keras.initializers import normal
except ImportError:  # tf 1.8.0 uses keras._impl directory
    from tensorflow.python.keras._impl.keras.initializers import normal

from deepcell.layers import Cast, Shape, UpsampleLike
from deepcell.layers import Upsample, RoiAlign, ConcatenateBoxes
from deepcell.layers import ClipBoxes, RegressBoxes, FilterDetections
from deepcell.layers import TensorProduct, ImageNormalization2D, Location2D
from deepcell.model_zoo.retinamovie import retinamovie
from deepcell.model_zoo.maskrcnn import default_mask_model, default_final_detection_model
from deepcell.model_zoo.retinamovie import __build_anchors
from deepcell.utils.retinanet_anchor_utils import AnchorParameters
from deepcell.utils.backbone_utils import get_backbone

class RoiAlign(Layer):
   # Modified ROI Align layer
   # Only takes in one feature map
   # Feature map must be the size of the original image
    def __init__(self, crop_size=(14, 14), parallel_iterations=32, **kwargs):
        self.crop_size = crop_size
        self.parallel_iterations = parallel_iterations        
        super(RoiAlign, self).__init__(**kwargs)    
        
    def call(self, inputs, **kwargs):
        boxes = K.stop_gradient(inputs[0])
        fpn = K.stop_gradient(inputs[1])
        
        if K.ndim(boxes) == 4:
            time_distributed = True
        else:
            time_distributed = False
            
        if time_distributed:
            boxes_shape = K.shape(boxes)
            fpn_shape = K.shape(fpn)
            
            new_boxes_shape =  [-1] + [boxes_shape[i] for i in range(2, K.ndim(boxes))]
            new_fpn_shape = [-1] + [fpn_shape[i] for i in range(2, K.ndim(fpn))]
            
            boxes = K.reshape(boxes, new_boxes_shape)
            fpn = K.reshape(fpn, new_fpn_shape)
            
        image_shape = K.cast(K.shape(fpn), K.floatx())        
        
        def _roi_align(args):
            boxes = args[0]
            fpn = args[1]            # process the feature map
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]            
            
            fpn_shape = K.cast(K.shape(fpn), dtype=K.floatx())
            norm_boxes = K.stack([
               (y1 / image_shape[1] * fpn_shape[0]) / (fpn_shape[0] - 1),
               (x1 / image_shape[2] * fpn_shape[1]) / (fpn_shape[1] - 1),
               (y2 / image_shape[1] * fpn_shape[0] - 1) / (fpn_shape[0] - 1),
               (x2 / image_shape[2] * fpn_shape[1] - 1) / (fpn_shape[1] - 1)
               ], axis=1)            
            
            rois = tf.image.crop_and_resize(
               K.expand_dims(fpn, axis=0),
               norm_boxes,
               tf.zeros((K.shape(norm_boxes)[0],), dtype='int32'),
               self.crop_size)            
            
            return rois        
        
        roi_batch = tf.map_fn(
           _roi_align,
           elems=[boxes, fpn],
           dtype=K.floatx(),
           parallel_iterations=self.parallel_iterations)
        
        if time_distributed:
            roi_shape = tf.shape(roi_batch)
            new_roi_shape = [boxes_shape[0], boxes_shape[1]] + [roi_shape[i] for i in range(1, K.ndim(roi_batch))]
            roi_batch = tf.reshape(roi_batch, new_roi_shape)
            
        return roi_batch
        
    def compute_output_shape(self, input_shape):
        if len(input_shape[3]) == 4:
            output_shape = [
               input_shape[1][0],
               None,
               self.crop_size[0],
               self.crop_size[1],
               input_shape[3][-1]
           ]
            return tensor_shape.TensorShape(output_shape)    
    
    def get_config(self):
        config = {'crop_size': self.crop_size}
        base_config = super(RoiAlign, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))