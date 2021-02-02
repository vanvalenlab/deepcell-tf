"""Tests for the convolutional recurrent layers"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.framework import test_util as tf_test_util

from deepcell import layers


class ConvGRU2DTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            data_format=['channels_first', 'channels_last'],
            return_sequences=[True, False]))
    def test_conv_gru_2d(self, data_format, return_sequences):
        num_row = 3
        num_col = 3
        filters = 2
        num_samples = 1
        input_channel = 2
        input_num_row = 5
        input_num_col = 5
        sequence_len = 2

        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, sequence_len,
                                    input_channel,
                                    input_num_row, input_num_col)
        else:
            inputs = np.random.rand(num_samples, sequence_len,
                                    input_num_row, input_num_col,
                                    input_channel)

        # test for return state:
        x = tf.keras.layers.Input(batch_shape=inputs.shape)
        kwargs = {'data_format': data_format,
                  'return_sequences': return_sequences,
                  'return_state': True,
                  'stateful': True,
                  'filters': filters,
                  'kernel_size': (num_row, num_col),
                  'padding': 'valid'}
        layer = layers.ConvGRU2D(**kwargs)
        layer.build(inputs.shape)

        outputs = layer(x)
        _, states = outputs[0], outputs[1:]
        self.assertEqual(len(states), len(layer.cell.state_size))
        model = tf.keras.models.Model(x, states[0])
        state = model.predict(inputs)

        self.assertAllClose(
            tf.keras.backend.eval(layer.states[0]), state, atol=1e-4)

        # test for output shape:
        custom_objects = {'ConvGRU2D': layers.ConvGRU2D}
        with tf.keras.utils.custom_object_scope(custom_objects):
            testing_utils.layer_test(
                layers.ConvGRU2D,
                kwargs={'data_format': data_format,
                        'return_sequences': return_sequences,
                        'filters': filters,
                        'kernel_size': (num_row, num_col),
                        'padding': 'valid'},
                input_shape=inputs.shape)

    def test_conv_gru_2d_statefulness(self):
        # Tests for statefulness
        num_row = 3
        num_col = 3
        filters = 2
        num_samples = 1
        input_channel = 2
        input_num_row = 5
        input_num_col = 5
        sequence_len = 2
        inputs = np.random.rand(num_samples, sequence_len,
                                input_num_row, input_num_col,
                                input_channel)

        with self.cached_session():
            model = tf.keras.models.Sequential()
            kwargs = {'data_format': 'channels_last',
                      'return_sequences': False,
                      'filters': filters,
                      'kernel_size': (num_row, num_col),
                      'stateful': True,
                      'batch_input_shape': inputs.shape,
                      'padding': 'same'}
            layer = layers.ConvGRU2D(**kwargs)

            model.add(layer)
            model.compile(optimizer='sgd', loss='mse')
            out1 = model.predict(np.ones_like(inputs))

            # train once so that the states change
            model.train_on_batch(np.ones_like(inputs),
                                 np.random.random(out1.shape))
            out2 = model.predict(np.ones_like(inputs))

            # if the state is not reset, output should be different
            self.assertNotEqual(out1.max(), out2.max())

            # check that output changes after states are reset
            # (even though the model itself didn't change)
            layer.reset_states()
            out3 = model.predict(np.ones_like(inputs))
            self.assertNotEqual(out3.max(), out2.max())

            # check that container-level reset_states() works
            model.reset_states()
            out4 = model.predict(np.ones_like(inputs))
            self.assertAllClose(out3, out4, atol=1e-5)

            # check that the call to `predict` updated the states
            out5 = model.predict(np.ones_like(inputs))
            self.assertNotEqual(out4.max(), out5.max())

    def test_conv_gru_2d_regularizers(self):
        # check regularizers
        num_row = 3
        num_col = 3
        filters = 2
        num_samples = 1
        input_channel = 2
        input_num_row = 5
        input_num_col = 5
        sequence_len = 2
        inputs = np.random.rand(num_samples, sequence_len,
                                input_num_row, input_num_col,
                                input_channel)

        with self.cached_session():
            kwargs = {'data_format': 'channels_last',
                      'return_sequences': False,
                      'kernel_size': (num_row, num_col),
                      'stateful': True,
                      'filters': filters,
                      'batch_input_shape': inputs.shape,
                      'kernel_regularizer':
                          tf.keras.regularizers.L1L2(l1=0.01),
                      'recurrent_regularizer':
                          tf.keras.regularizers.L1L2(l1=0.01),
                      'activity_regularizer': 'l2',
                      'bias_regularizer': 'l2',
                      'kernel_constraint': 'max_norm',
                      'recurrent_constraint': 'max_norm',
                      'bias_constraint': 'max_norm',
                      'padding': 'same'}

            layer = layers.ConvGRU2D(**kwargs)
            layer.build(inputs.shape)
            self.assertEqual(len(layer.losses), 3)
            layer(tf.keras.backend.variable(np.ones(inputs.shape)))
            self.assertEqual(len(layer.losses), 4)

    def test_conv_gru_2d_dropout(self):
        # check dropout
        with self.cached_session():
            custom_objects = {'ConvGRU2D': layers.ConvGRU2D}
            with tf.keras.utils.custom_object_scope(custom_objects):
                testing_utils.layer_test(
                    layers.ConvGRU2D,
                    kwargs={'data_format': 'channels_last',
                            'return_sequences': False,
                            'filters': 2,
                            'kernel_size': (3, 3),
                            'padding': 'same',
                            'dropout': 0.1,
                            'recurrent_dropout': 0.1},
                    input_shape=(1, 2, 5, 5, 2))

    def test_conv_gru_2d_cloning(self):
        with self.cached_session():
            model = tf.keras.models.Sequential()
            model.add(layers.ConvGRU2D(5, 3, input_shape=(None, 5, 5, 3)))

            test_inputs = np.random.random((2, 4, 5, 5, 3))
            reference_outputs = model.predict(test_inputs)
            weights = model.get_weights()

        # Use a new graph to clone the model
        with self.cached_session():
            clone = tf.keras.models.clone_model(model)
            clone.set_weights(weights)

            outputs = clone.predict(test_inputs)
            self.assertAllClose(reference_outputs, outputs, atol=1e-5)
