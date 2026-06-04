"""Tests for custom loss functions"""


import numpy as np

from tensorflow import keras
from tensorflow.python.platform import test

from deepcell import losses


ALL_LOSSES = [
    losses.categorical_crossentropy,
    losses.weighted_categorical_crossentropy,
    losses.sample_categorical_crossentropy,
    losses.weighted_focal_loss,
    losses.smooth_l1,
    losses.focal,
    # losses.dice_loss,
    # losses.discriminative_instance_loss
]


class KerasLossesTest(test.TestCase):

    def test_objective_shapes_3d(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.random((5, 6, 7)))
            y_b = keras.backend.variable(np.random.random((5, 6, 7)))
            for obj in ALL_LOSSES:
                objective_output = obj(y_a, y_b)
                self.assertListEqual(objective_output.shape.as_list(), [5, 6])

    def test_objective_shapes_2d(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.random((6, 7)))
            y_b = keras.backend.variable(np.random.random((6, 7)))
            for obj in ALL_LOSSES:
                objective_output = obj(y_a, y_b)
                self.assertListEqual(objective_output.shape.as_list(), [6])


if __name__ == '__main__':
    test.main()
