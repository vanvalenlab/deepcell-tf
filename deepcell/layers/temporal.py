"""Upsampling layers"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Comparison(Layer):
    """Layer for comparing two sequences of inputs."""
    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]

        x = tf.expand_dims(x, 3)
        multiples = [1, 1, 1, tf.shape(y)[2], 1]
        x = tf.tile(x, multiples)

        y = tf.expand_dims(y, 2)
        multiples = [1, 1, tf.shape(x)[2], 1, 1]
        y = tf.tile(y, multiples)

        return tf.concat([x, y], axis=-1)


class DeltaReshape(Layer):
    """Reshape changes between current and future frames"""
    def call(self, inputs):
        current = inputs[0]
        future = inputs[1]
        current = tf.expand_dims(current, axis=3)
        multiples = [1, 1, 1, tf.shape(future)[2], 1]
        output = tf.tile(current, multiples)
        return output


class Unmerge(Layer):
    """Unmerge temporal inputs"""
    def __init__(self, track_length, max_cells, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.track_length = track_length
        self.max_cells = max_cells
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        new_shape = [-1, self.track_length, self.max_cells, self.embedding_dim]
        output = tf.reshape(inputs, new_shape)
        return output

    def get_config(self):
        config = {
            'track_length': self.track_length,
            'max_cells': self.max_cells,
            'embedding_dim': self.embedding_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TemporalMerge(Layer):
    """Layer for merging the time dimension of a Tensor.

    Args:
        encoder_dim (int): desired encoder dimension.
    """
    def __init__(self, encoder_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.lstm = tf.keras.layers.LSTM(
            self.encoder_dim,
            return_sequences=True,
            name=f'{self.name}_lstm')

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        # reshape away the temporal axis
        x = tf.reshape(inputs, [-1, input_shape[1], self.encoder_dim])
        x = self.lstm(x)
        output_shape = [-1, input_shape[1], input_shape[2], self.encoder_dim]
        x = tf.reshape(x, output_shape)
        return x

    def get_config(self):
        config = {
            'encoder_dim': self.encoder_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
