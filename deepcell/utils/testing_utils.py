# Copyright 2016-2019 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Utilities testing Keras layers"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import threading

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.util import tf_inspect


def layer_test(layer_cls, kwargs=None, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None, custom_objects=None):
    """Test routine for a layer with a single input and single output.

    Args:
        layer_cls: Layer class object.
        kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
        input_shape: Input shape tuple.
        input_dtype: Data type of the input data.
        input_data: Numpy array of input data.
        expected_output: Shape tuple for the expected shape of the output.
        expected_output_dtype: Data type expected for the output.
        custom_objects: Custom Objects to test custom layers

    Returns:
        The output data (Numpy array) returned by the layer, for additional
        checks to be done by the calling code.
    """
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = 'float32'
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = 10 * np.random.random(input_data_shape)
        if input_dtype[:5] == 'float':
            input_data -= 0.5
        input_data = input_data.astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape
    if input_dtype is None:
        input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
        kwargs['weights'] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    x = keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    if keras.backend.dtype(y) != expected_output_dtype:
        raise AssertionError(
            'When testing layer %s, for input %s, found output dtype=%s but '
            'expected to find %s.\nFull kwargs: %s' %
            (layer_cls.__name__,
             x,
             keras.backend.dtype(y),
             expected_output_dtype,
             kwargs))
    # check shape inference
    model = keras.models.Model(x, y)
    expected_output_shape = tuple(
        layer.compute_output_shape(
            tensor_shape.TensorShape(input_shape)).as_list())
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(expected_output_shape,
                                        actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    'When testing layer %s, for input %s, found output_shape='
                    '%s but expected to find %s.\nFull kwargs: %s' %
                    (layer_cls.__name__,
                     x,
                     actual_output_shape,
                     expected_output_shape,
                     kwargs))
    if expected_output is not None:
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    # Edited to add custom_objects to model.from_config
    recovered_model = keras.models.Model.from_config(
        model_config, custom_objects=custom_objects)
    # End Edits
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=1e-3)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # train(). This was causing some error for layer with Defun as it body.
    # See b/120160788 for more details. This should be mitigated after 2.0.
    model = keras.models.Model(x, layer(x))
    if _thread_local_data.run_eagerly is not None:
        model.compile(
            'rmsprop',
            'mse',
            weighted_metrics=['acc'],
            run_eagerly=should_run_eagerly())
    else:
        model.compile('rmsprop', 'mse', weighted_metrics=['acc'])
    model.train_on_batch(input_data, actual_output)

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)

    model = keras.models.Sequential()
    model.add(layer)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(expected_output_shape,
                                        actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    'When testing layer %s, for input %s, found output_shape='
                    '%s but expected to find %s.\nFull kwargs: %s' %
                    (layer_cls.__name__,
                     x,
                     actual_output_shape,
                     expected_output_shape,
                     kwargs))
    if expected_output is not None:
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    # Edited to add custom_objects to model.from_config
    recovered_model = keras.models.Sequential.from_config(
        model_config, custom_objects=custom_objects)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=1e-3)

    # for further checks in the caller function
    return actual_output


_thread_local_data = threading.local()
_thread_local_data.model_type = None
_thread_local_data.run_eagerly = None


def should_run_eagerly():
    """Returns whether the models we are testing should be run eagerly."""
    if _thread_local_data.run_eagerly is None:
        raise ValueError('Cannot call `should_run_eagerly()` outside of a '
                         '`run_eagerly_scope()` or `run_all_keras_modes` '
                         'decorator.')

    return _thread_local_data.run_eagerly and context.executing_eagerly()
