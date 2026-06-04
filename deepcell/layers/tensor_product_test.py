"""Tests for the tensor product layers"""

import numpy as np
import tensorflow as tf

from keras import keras_parameterized
from keras import testing_utils
from tensorflow.python.platform import test

from deepcell import layers


@keras_parameterized.run_all_keras_modes
class TensorProdTest(keras_parameterized.TestCase):

    def test_tensorproduct(self):
        custom_objects = {'TensorProduct': layers.TensorProduct}
        with tf.keras.utils.custom_object_scope(custom_objects):
            testing_utils.layer_test(
                layers.TensorProduct,
                kwargs={'output_dim': 3},
                input_shape=(3, 2))

            testing_utils.layer_test(
                layers.TensorProduct,
                kwargs={'output_dim': 3},
                input_shape=(3, 4, 2))

            testing_utils.layer_test(
                layers.TensorProduct,
                kwargs={'output_dim': 3},
                input_shape=(None, None, 2))

            testing_utils.layer_test(
                layers.TensorProduct,
                kwargs={'output_dim': 3},
                input_shape=(3, 4, 5, 2))

            testing_utils.layer_test(
                layers.TensorProduct,
                kwargs={'output_dim': 3,
                        'data_format': 'channels_first'},
                input_shape=(3, 2, 4, 5))

            # test no bias
            testing_utils.layer_test(
                layers.TensorProduct,
                kwargs={'output_dim': 2,
                        'use_bias': False},
                input_shape=(3, 5, 6, 4))

            # test bad input channel
            with self.assertRaises(ValueError):
                testing_utils.layer_test(
                    layers.TensorProduct,
                    kwargs={'output_dim': 3},
                    input_shape=(3, 5, 6, None))

    def test_tensorproduct_regularization(self):
        layer = layers.TensorProduct(
            3,
            kernel_regularizer=tf.keras.regularizers.l1(0.01),
            bias_regularizer='l1',
            activity_regularizer='l2',
            name='tensorproduct_reg')
        layer(tf.keras.backend.variable(np.ones((2, 4))))
        self.assertEqual(3, len(layer.losses))

    def test_tensorproduct_constraints(self):
        k_constraint = tf.keras.constraints.max_norm(0.01)
        b_constraint = tf.keras.constraints.max_norm(0.01)
        layer = layers.TensorProduct(
            3,
            kernel_constraint=k_constraint,
            bias_constraint=b_constraint)
        layer(tf.keras.backend.variable(np.ones((2, 4))))
        self.assertEqual(layer.kernel.constraint, k_constraint)
        self.assertEqual(layer.bias.constraint, b_constraint)


if __name__ == '__main__':
    test.main()
