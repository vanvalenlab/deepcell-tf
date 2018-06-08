"""
dc_running_functions.py

Functions for exporting convolutional neural networks
written in Keras to Tensorflow serving

@author: David Van Valen
"""

from __future__ import print_function
from __future__ import division

import os
import warnings

import numpy as np
import tifffile as tiff
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder 
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

from .dc_helper_functions import get_images_from_directory, process_image
from .dc_settings import CHANNELS_FIRST, CHANNELS_LAST

FLAGS = tf.app.flags.FLAGS

def export_model(keras_model, export_path, model_version = 0, weights_path = None):
    
    # Start the tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K._LEARNING_PHASE = tf.constant(0)
    K.set_learning_phase(0)

    # Load the model and the weights
    if weights_path is not None:
        keras_model.load_weights(weights_path)

    # Define prediction signature
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"image": keras_model.input}, {"prediction":keras_model.output})

    # Create export path if it doesn't exist
    export_path = os.path.join(export_path, str(model_version))
    builder = saved_model_builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    #Initialize global variables and the model
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Add the meta_graph and the variables to the builder
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: 
                prediction_signature
        },
        legacy_init_op=legacy_init_op)

    # Save the graph
    builder.save()


