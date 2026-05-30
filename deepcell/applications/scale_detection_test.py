"""Tests for ScaleDetectionModel"""


import numpy as np
import tensorflow as tf

from deepcell.applications.scale_detection import ScaleDetectionModel
from deepcell.applications import ScaleDetection


class TestScaleDetectionModel(tf.test.TestCase):

    def test_scale_detection_model(self):

        valid_backbones = ['featurenet']
        input_shape = (216, 216, 1)  # channels will be set to 3

        batch_shape = tuple([8] + list(input_shape))

        X = np.random.random(batch_shape)

        for backbone in valid_backbones:
            with self.cached_session():
                inputs = tf.keras.layers.Input(shape=input_shape)
                model = ScaleDetectionModel(
                    inputs=inputs,
                    backbone=backbone)

                y = model.predict(X)

                assert len(y.shape) == 2
                assert y.shape[0] == X.shape[0]

            with self.cached_session():
                model = ScaleDetectionModel(
                    input_shape=input_shape,
                    backbone=backbone)

                y = model.predict(X)

                assert len(y.shape) == 2
                assert y.shape[0] == X.shape[0]


class TestScaleDetection(tf.test.TestCase):

    def test_scale_detection_app(self):
        with self.cached_session():
            model = ScaleDetectionModel(input_shape=(128, 128, 1))
            app = ScaleDetection(model)

            # test output shape
            shape = app.model.output_shape
            self.assertEqual(len(shape), 2)
            self.assertEqual(shape[-1], 1)

            # test predict with default
            x = np.random.rand(1, 500, 500, 1)
            y = app.predict(x)
            self.assertIsInstance(y, np.float32)
