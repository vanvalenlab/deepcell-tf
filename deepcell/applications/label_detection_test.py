"""Tests for LabelDetectionModel"""


import numpy as np
import tensorflow as tf

from deepcell.applications.label_detection import LabelDetectionModel
from deepcell.applications import LabelDetection


class TestLabelDetectionModel(tf.test.TestCase):

    def test_label_detection_model(self):

        valid_backbones = ['featurenet']
        input_shape = (216, 216, 1)  # channels will be set to 3

        batch_shape = tuple([8] + list(input_shape))

        X = np.random.random(batch_shape)

        for backbone in valid_backbones:
            with self.cached_session():
                inputs = tf.keras.layers.Input(shape=input_shape)
                model = LabelDetectionModel(
                    inputs=inputs,
                    backbone=backbone)

                y = model.predict(X)

                assert len(y.shape) == 2
                assert y.shape[0] == X.shape[0]

            with self.cached_session():
                model = LabelDetectionModel(
                    input_shape=input_shape,
                    backbone=backbone)

                y = model.predict(X)

                assert len(y.shape) == 2
                assert y.shape[0] == X.shape[0]


class TestLabelDetection(tf.test.TestCase):

    def test_label_detection_app(self):
        with self.cached_session():
            num_classes = 3
            model = LabelDetectionModel(input_shape=(128, 128, 1))
            app = LabelDetection(model)

            # test output shape
            shape = app.model.output_shape
            self.assertEqual(len(shape), 2)
            self.assertEqual(shape[-1], num_classes)

            # test predict with default
            x = np.random.rand(1, 500, 500, 1)
            y = app.predict(x)

            self.assertTrue(int(y) in set(range(num_classes)))
