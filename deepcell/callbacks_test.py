"""Custom Callbacks for DeepCell"""


import sys

import tensorflow as tf
from keras import keras_parameterized
from keras import testing_utils

from deepcell import callbacks


class TestInferenceTimer(keras_parameterized.TestCase):
    """Callback to log inference speed per epoch."""

    @keras_parameterized.run_all_keras_modes
    def test_inference_time_logging(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1))
        model.compile(
            'sgd',
            loss='mse',
            run_eagerly=testing_utils.should_run_eagerly())

        x = tf.ones((200, 3))
        y = tf.zeros((200, 2))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)
        expected_log = r'(.*Average inference.*)+'

        cbks = [callbacks.InferenceTimer()]

        with self.captureWritesToStream(sys.stdout) as printed:
            y = model.call(x)
            model.fit(dataset, epochs=2, steps_per_epoch=10, callbacks=cbks)
            self.assertRegex(printed.contents(), expected_log)
