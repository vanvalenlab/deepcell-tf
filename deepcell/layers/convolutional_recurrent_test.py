"""Tests for the convolutional recurrent layers"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pytest
import numpy as np
from numpy.testing import assert_allclose

from tensorflow.python.keras import keras_parameterized

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell.utils import testing_utils
from deepcell import layers


class ConvolutionalRecurrentTest(keras_parameterized.TestCase):
    
    @keras_parameterized.run_all_keras_modes
    def test_convolutional_gru_2d(self):
        num_row = 3
        num_col = 3
        filters = 2
        num_samples = 1
        input_channel = 2
        input_num_row = 5
        input_num_col = 5
        sequence_len = 2
    
        custom_objects = {'ConvGRU2D': layers.ConvGRU2D}
        # custom_objects = {'ConvLSTM2D': convolutional_recurrent.ConvLSTM2D}
        
        for data_format in ['channels_first', 'channels_last']:
            if data_format == 'channels_first':
                inputs = np.random.rand(num_samples, sequence_len,
                                        input_channel,
                                        input_num_row, input_num_col)
            else:
                inputs = np.random.rand(num_samples, sequence_len,
                                        input_num_row, input_num_col,
                                        input_channel)
            for return_sequences in [True, False]:
                x = Input(batch_shape=inputs.shape)
                kwargs = {'data_format': data_format,
                          'return_sequences': return_sequences,
                          'return_state': True,
                          'stateful': True,
                          'filters': filters,
                          'kernel_size': (num_row, num_col),
                          'padding': 'valid'}
                #layer = convolutional_recurrent.ConvLSTM2D(**kwargs) 
                layer = layers.ConvGRU2D(**kwargs)
                layer.build(inputs.shape)
                
                outputs = layer(x)
                output, states = outputs[0], outputs[1:]
                
                assert len(states) == 2
                model = Model(x, states[0])
                state = model.predict(inputs)
                np.testing.assert_allclose(K.eval(layer.states[0]), state, atol=1e-4)

                # test for output shape:
                # output = testing_utils.layer_test(convolutional_recurrent.ConvLSTM2D,
                output = testing_utils.layer_test(layers.ConvGRU2D,
                        kwargs={'data_format': data_format,
                                'return_sequences': return_sequences,
                                'filters': filters,
                                'kernel_size': (num_row, num_col),
                                'padding': 'valid'},
                        custom_objects=custom_objects,
                        input_shape=inputs.shape)
  
                        
