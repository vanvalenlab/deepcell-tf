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
"""Applications objects for segmentation models"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.contrib import predictor

class TFSavedModel(object):
    def __init__(self,
            model_path):
        self.model = tf.compat.v2.keras.models.load_model(model_path)
        self.predict_fn = predictor.from_saved_model(model_path)

    def predict(self, X, batch_size=1):
        output_keys = self.model.signatures['serving_default'].structured_outputs.keys()
        output_keys = sorted(output_keys)
        output_list = [[] for key in output_keys]
        for i in range(0, X.shape[0], batch_size):
            X_part = X[i:min(i+batch_size, X.shape[0])]
            outputs = self.predict_fn({'image': X_part})
            for key, o in zip(output_keys, output_list):
                o.append(outputs[key])

        output_list = [np.concatenate(o, axis=0) for o in output_list]

        return output_list

class TFLiteModel(object):
    def __init__(self, 
            model_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def _norm_images(self, images):
        mean = np.mean(images, axis=(1,2), keepdims=True)
        std = np.std(images, axis=(1,2), keepdims=True)
        norm = (images-mean)/std
        return norm

    def _add_location(self, images):
        x = np.arange(0, images.shape[1], dtype='float32')
        y = np.arange(0, images.shape[2], dtype='float32')

        x = x/max(x)
        y = y/max(y)

        loc_x, loc_y = np.meshgrid(x, y, indexing='ij')
        loc = np.stack([loc_x, loc_y], axis=-1)
        loc = np.expand_dims(loc, axis=0)
        loc = np.tile(loc, (images.shape[0],1,1,1))
        images_with_loc = np.concatenate([images, loc], axis=-1)
        return images_with_loc

    def _quantize(self, data):
        shape = self.input_details[0]['shape']
        dtype = self.input_details[0]['dtype']
        a, b = self.input_details[0]['quantization']
        
        quantized_data = (data/a + b).astype(dtype).reshape(shape)

        return quantized_data

    def _dequantize(self, output_list):
        dequantized_list = []
        for i, output in enumerate(output_list):
            a, b = self.output_details[i]['quantization']
            dequantized_output = (output - b)*a
            dequantized_list.append(dequantized_output)

        return dequantized_list

    def predict(self, X, batch_size=1):
        output_list = [[] for o in self.output_details]
        X = self._norm_images(X)
        X = self._add_location(X)
        for i in range(0, X.shape[0], batch_size):
            X_part = X[i:min(i+batch_size, X.shape[0])]

            # Quantize
            X_part_quantize = self._quantize(X_part)
            
            # Run model
            self.interpreter.set_tensor(self.input_details[0]['index'], X_part_quantize)
            self.interpreter.invoke()
            
            output = [self.interpreter.get_tensor(detail['index']) for detail in self.output_details]
            output_float = self._dequantize(output)

            for o, o_float in zip(output_list, output_float):
                o.append(o_float)

        output_list = [np.concatenate(o, axis=0) for o in output_list]

        return output_list


class SegmentationApplication(object):
    def __init__(self,
            model,
            weights_path=None,
            model_image_shape=(128,128,1),
            dataset_metadata=None,
            model_metadata=None,
            model_mpp=0.65,
            preprocessing_fn=None,
            postprocessing_fn=None):

        self.model = model

        if weights_path is not None:
            self.model.load_weights(weights_path)

        self.model_image_shape = model_image_shape
        self.model_mpp = model_mpp
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn

    def _tile_image(self, image):
        image_size_x, image_size_y = image.shape[1:3]
        tile_size_x = self.model_image_shape[0]
        tile_size_y = self.model_image_shape[1]

        stride_x = np.int(0.75 * tile_size_x)
        stride_y = np.int(0.75 * tile_size_y)

        rep_number_x = np.int(np.ceil((image_size_x - tile_size_x)/stride_x + 1))
        rep_number_y = np.int(np.ceil((image_size_y - tile_size_y)/stride_y + 1))
        new_batch_size = image.shape[0] * rep_number_x * rep_number_y

        tiles_shape = (new_batch_size, tile_size_x, tile_size_y, image.shape[3])
        tiles = np.zeros(tiles_shape, dtype=K.floatx())

        counter = 0
        batches = []
        x_starts = []
        x_ends = []
        y_starts = []
        y_ends = []

        for b in range(image.shape[0]):
            for i in range(rep_number_x):
                for j in range(rep_number_y):
                    _axis = 1
                    if i != rep_number_x - 1:
                        x_start, x_end = i*stride_x, i*stride_x + tile_size_x
                    else:
                        x_start, x_end = -tile_size_x, image.shape[_axis]

                    if j != rep_number_y - 1:
                        y_start, y_end = j*stride_y, j*stride_y + tile_size_y
                    else:
                        y_start, y_end = -tile_size_y, image.shape[_axis + 1]

                    tiles[counter] = image[b, x_start:x_end, y_start:y_end, :]
                    batches.append(b)
                    x_starts.append(x_start)
                    x_ends.append(x_end)
                    y_starts.append(y_start)
                    y_ends.append(y_end)
                    counter += 1

        tiles_info = {}
        tiles_info['batches'] = batches
        tiles_info['x_starts'] = x_starts
        tiles_info['x_ends'] = x_ends
        tiles_info['y_starts'] = y_starts
        tiles_info['y_ends'] = y_ends
        tiles_info['stride_x'] = stride_x
        tiles_info['stride_y'] = stride_y
        tiles_info['image_shape'] = image.shape

        return tiles, tiles_info

    def _untile_image(self, tiles, tiles_info):
        _axis = 1
        image_shape = tiles_info['image_shape']
        batches = tiles_info['batches']
        x_starts = tiles_info['x_starts']
        x_ends = tiles_info['x_ends']
        y_starts = tiles_info['y_starts']
        y_ends = tiles_info['y_ends']
        stride_x = tiles_info['stride_x']
        stride_y = tiles_info['stride_y']

        tile_size_x = self.model_image_shape[0]
        tile_size_y = self.model_image_shape[1]

        image_shape = [image_shape[0], image_shape[1], image_shape[2], tiles.shape[-1]]
        image = np.zeros(image_shape, dtype = K.floatx())
        n_tiles = tiles.shape[0]

        for tile, batch, x_start, x_end, y_start, y_end in zip(tiles, batches, x_starts, x_ends, y_starts, y_ends):
            tile_x_start = 0
            tile_x_end = tile_size_x
            tile_y_start = 0
            tile_y_end = tile_size_y

            if x_start != 0:
                x_start += (tile_size_x - stride_x)/2
                tile_x_start += (tile_size_x - stride_x)/2
            if x_end != image_shape[_axis]:
                x_end -= (tile_size_x - stride_x)/2
                tile_x_end -= (tile_size_x - stride_x)/2
            if y_start != 0:
                y_start += (tile_size_y - stride_y)/2
                tile_y_start += (tile_size_y - stride_y)/2
            if y_end != image_shape[_axis]:
                y_end -= (tile_size_y - stride_y)/2
                tile_y_end -= (tile_size_y - stride_y)/2

            x_start = np.int(x_start)
            x_end = np.int(x_end)
            y_start = np.int(y_start)
            y_end = np.int(y_end)

            tile_x_start = np.int(tile_x_start)
            tile_x_end = np.int(tile_x_end)
            tile_y_start = np.int(tile_y_start)
            tile_y_end = np.int(tile_y_end)

            image[batch, x_start:x_end, y_start:y_end, :] = tile[tile_x_start:tile_x_end, tile_y_start:tile_y_end, :]

        return image
        
    def predict(self, image, 
                batch_size=4,
                image_mpp=None, 
                preprocess_kwargs={}, 
                postprocess_kwargs={}):

        # Resize image if necessary
        
        # Preprocess image
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)

        # Tile images 
        tiles, tiles_info = self._tile_image(image)

        # Run images through model
        output_tiles = self.model.predict(tiles, batch_size=batch_size)

        if not isinstance(output_tiles, list):
            output_tiles = [output_tiles]

        # Untile images
        output_images = [self._untile_image(o, tiles_info) for o in output_tiles]

        # Postprocess predictions to create label image
        label_image = self.postprocessing_fn(output_images, **postprocess_kwargs)

        # Resize label_image back to original resolution if necessary
        
        return image, tiles, label_image, output_tiles, output_images