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

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.builder import SavedModelBuilder


def export_model(keras_model, export_path, model_version=0, weights_path=None):
    """Export a model for use with tensorflow-serving.

    Args:
        keras_model: instantiated Keras model to export
        export_path: destination to save the exported model files
        model_version: integer version of the model
        weights_path: path to a .h5 or .tf weights file for the model to load
    """
    # Start the tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
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

    if type(keras_model.input) is list:
        output = keras_model.output[-1]
    else:
        output = keras_model.output

    # Define prediction signature
    if type(keras_model.input) is list:
        input_map = {"input{}".format(i): input_tensor
                     for i, input_tensor in enumerate(keras_model.input)}
        output_map = {"prediction": output}
    else:
        input_map = {"input": keras_model.input}
        output_map = {"prediction": output}

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
