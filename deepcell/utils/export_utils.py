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
"""Save Keras models as a SavedModel for TensorFlow Serving"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import tf_logging


def export_model(keras_model, export_path, model_version=0, weights_path=None,
                 include_optimizer=True, overwrite=True, save_format='tf'):
    """Export a model for use with TensorFlow Serving.

    DEPRECATED: ``tf.keras.models.save_model`` is preferred.

    Args:
        keras_model (tensorflow.keras.Model): Instantiated Keras model.
        export_path (str): Destination to save the exported model files.
        model_version (int): Integer version of the model.
        weights_path (str): Path to a ``.h5`` or ``.tf`` weights file.
        include_optimizer (bool): Whether to export the optimizer.
        overwrite (bool): Whether to overwrite any existing files in
            ``export_path``.
        save_format (str): Saved model format, one of ``'tf'`` or ``'h5'``.
    """
    tf_logging.warn('`export_model` is deprecated. '
                    'Please use `tf.keras.models.save_model` instead.')

    if weights_path:
        keras_model.load_weights(weights_path, by_name=True)

    tf.keras.models.save_model(
        keras_model,
        os.path.join(export_path, str(int(model_version))),
        overwrite=overwrite,
        include_optimizer=include_optimizer,
        save_format=save_format,
        signatures=None,
    )


def export_model_to_tflite(model_file, export_path, calibration_images,
                           norm=True, location=True, file_name='model.tflite'):
    """Export a saved keras model to tensorflow-lite with int8 precision.

    This export function has only been tested with ``PanopticNet`` models.
    For the export to be successful, the ``PanopticNet`` model must have
    ``norm_method`` set to ``None``, ``location`` set to ``False``,
    and the upsampling layers must use ``bilinear`` interpolation.

    Args:
        model_file (str): Path to saved model file
        export_path (str): Directory to save the exported tflite model
        calibration_images (numpy.array): Array of images used for calibration
            during model quantization
        norm (bool): Whether to normalize calibration images.
        location (bool): Whether to append a location image
            to calibration images.
        file_name (str): File name for the exported model. Defaults to
            'model.tflite'
    """
    # Define helper function - normalization
    def norm_images(images):
        mean = np.mean(images, axis=(1, 2), keepdims=True)
        std = np.std(images, axis=(1, 2), keepdims=True)
        norm = (images - mean) / std
        return norm

    # Define helper function - add location layer
    def add_location(images):
        x = np.arange(0, images.shape[1], dtype='float32')
        y = np.arange(0, images.shape[2], dtype='float32')

        x = x / max(x)
        y = y / max(y)

        loc_x, loc_y = np.meshgrid(x, y, indexing='ij')
        loc = np.stack([loc_x, loc_y], axis=-1)
        loc = np.expand_dims(loc, axis=0)
        loc = np.tile(loc, (images.shape[0], 1, 1, 1))
        images_with_loc = np.concatenate([images, loc], axis=-1)
        return images_with_loc

    # Create generator to calibrate model quantization
    calibration_images = calibration_images.astype('float32')
    if norm:
        calibration_images = norm_images(calibration_images)
    if location:
        calibration_images = add_location(calibration_images)

    def representative_data_gen():
        for image in calibration_images:
            data = [np.expand_dims(image, axis=0)]
            yield data

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()

    # Save converted model
    save_path = os.path.join(export_path, file_name)
    open(save_path, "wb").write(tflite_quant_model)

    return tflite_model
