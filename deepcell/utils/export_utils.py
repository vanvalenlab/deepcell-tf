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
"""Functions for exporting convolutional neural networks for TF serving"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.builder import SavedModelBuilder


def export_model(keras_model, export_path, model_version=0, weights_path=None):
    """Export a model for use with tensorflow-serving.

    Args:
        keras_model (tensorflow.keras.Model): instantiated Keras model to export
        export_path (str): destination to save the exported model files
        model_version (int): integer version of the model
        weights_path (str): path to a .h5 or .tf weights file
    """
    # Start the tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,
                                allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K._LEARNING_PHASE = tf.constant(0)
    K.set_learning_phase(0)

    # Create export path if it doesn't exist
    export_path = os.path.join(export_path, str(model_version))
    builder = SavedModelBuilder(export_path)
    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    # Initialize global variables and the model
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Load the model and the weights
    if weights_path is not None:
        keras_model.load_weights(weights_path)

    # Export for tracking
    if isinstance(keras_model.input, list):
        input_map = {"input{}".format(i): input_tensor
                     for i, input_tensor in enumerate(keras_model.input)}
        output_map = {'output': keras_model.output}
    # Export for panoptic
    elif isinstance(keras_model.output, list):
        input_map = {'image': keras_model.input}
        output_map = {'prediction{}'.format(i): tensor
                      for i, tensor in enumerate(keras_model.output)}
    # Export for normal model architectures
    else:
        input_map = {"image": keras_model.input}
        output_map = {"prediction": keras_model.output}

    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        input_map,
        output_map
    )

    # Add the meta_graph and the variables to the builder
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        })

    # Save the graph
    builder.save()


def export_model_to_tflite(model_file, export_path, calibration_images,
                           norm=True, location=True, file_name='model.tflite'):
    """Export a saved keras model to tensorflow-lite with int8 precision.

    This export function has only been tested with PanopticNet models. For the
    export to be successful, the PanopticNet model must have normalization set
    to None, location set to False, and the upsampling layers must use bilinear
    interpolation.

    Args:
        model_file (str): Path to saved keras model
        export_path (str): Directory to save the exported tflite model
        calibration_images (numpy array): Array of images used for calibration
            during model quantization
        norm (boolean): Whether to normalize calibration images.
            Defaults to True.
        location (boolean): Whether to append a location image to calibration
            images. Defaults to True.
        file_name (str): File name for the exported model. Defaults to
            'model.tflite'
    """
    # Define helper functions
    def norm_images(images):
        mean = np.mean(images, axis=(1, 2), keepdims=True)
        std = np.std(images, axis=(1, 2), keepdims=True)
        norm = (images-mean)/std
        return norm

    def add_location(images):
        x = np.arange(0, images.shape[1], dtype='float32')
        y = np.arange(0, images.shape[2], dtype='float32')

        x = x/max(x)
        y = y/max(y)

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
