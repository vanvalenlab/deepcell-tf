"""Tests for running functions
"""

from absl.testing import parameterized

import numpy as np

from tensorflow import keras
from tensorflow.python.platform import test

from deepcell import layers
from deepcell import running


class RunningTests(test.TestCase, parameterized.TestCase):

    def test_get_cropped_input_shape(self):
        # test 2D images
        img_w, img_h = 30, 30
        batches = 5
        n_crops = 4
        field = 11
        # test channels_last
        keras.backend.set_image_data_format('channels_last')
        X = np.random.random((batches, img_w, img_h, 1))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field)

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (new_w, new_h, 1))

        # test channels_first
        X = np.random.random((batches, 1, img_w, img_h))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field,
            data_format='channels_first')

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (1, new_w, new_h))

        # test 3D images
        frames = 30

        # test channels_last
        X = np.random.random((batches, frames, img_w, img_h, 1))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field,
            data_format='channels_last')

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (frames, new_w, new_h, 1))

        # test channels_first
        X = np.random.random((batches, 1, frames, img_w, img_h))
        shape = running.get_cropped_input_shape(
            images=X,
            num_crops=n_crops,
            receptive_field=field,
            data_format='channels_first')

        new_w = img_w // n_crops + (field - 1)
        new_h = img_h // n_crops + (field - 1)
        self.assertEqual(shape, (1, frames, new_w, new_h))

    def test_get_padding_layers(self):
        # test reflection padding layer
        model = keras.models.Sequential()
        model.add(layers.ReflectionPadding2D(padding=(30, 30)))
        model.add(keras.layers.Dense(4))

        padded = running.get_padding_layers(model)
        self.assertEqual(len(padded), 1)

        # test zero padding layer
        model = keras.models.Sequential()
        model.add(keras.layers.ZeroPadding2D(padding=(30, 30)))
        model.add(keras.layers.Dense(4))

        padded = running.get_padding_layers(model)
        self.assertEqual(len(padded), 1)

        # test no padding layer
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(4))

        padded = running.get_padding_layers(model)
        self.assertEqual(len(padded), 0)

    @parameterized.named_parameters([
        {
            'testcase_name': '2d_channels_last',
            'data_format': 'channels_last',
            'shape': (2, 32, 32, 1)
        }, {
            'testcase_name': '3d_channels_last',
            'data_format': 'channels_last',
            'shape': (2, 8, 32, 32, 1)
        }, {
            'testcase_name': '2d_channels_first',
            'data_format': 'channels_first',
            'shape': (2, 1, 32, 32)
        }, {
            'testcase_name': '3d_channels_first',
            'data_format': 'channels_first',
            'shape': (2, 1, 8, 32, 32)
        },
    ])
    def test_process_whole_image(self, data_format, shape):
        keras.backend.set_image_data_format(data_format)

        num_crops = 2
        receptive_field = 3
        features = 3

        images = np.ones(shape)

        input_shape = running.get_cropped_input_shape(
            images, num_crops,
            receptive_field=receptive_field,
            data_format=data_format)

        for padding in ['reflect', 'zero']:
            with self.cached_session():
                inputs = keras.layers.Input(shape=input_shape)
                outputs = layers.TensorProduct(features)(inputs)
                model = keras.models.Model(inputs=inputs,
                                           outputs=[outputs, outputs])

                output = running.process_whole_image(
                    model, images,
                    num_crops=num_crops,
                    receptive_field=receptive_field,
                    padding=padding)

                if data_format == 'channels_first':
                    expected_shape = tuple([images.shape[0], features] +
                                           list(images.shape[2:]))
                else:
                    expected_shape = tuple([images.shape[0]] +
                                           list(images.shape[1:-1]) +
                                           [features])

                self.assertEqual(output.shape, expected_shape)

        with self.assertRaises(ValueError):
            inputs = keras.layers.Input(shape=(3, 4, 5))
            outputs = layers.TensorProduct(features)(inputs)
            model = keras.models.Model(inputs=inputs, outputs=outputs)

            output = running.process_whole_image(
                model, images,
                num_crops=num_crops,
                receptive_field=receptive_field,
                padding='reflect')

        with self.assertRaises(ValueError):
            inputs = keras.layers.Input(shape=input_shape)
            outputs = layers.TensorProduct(features)(inputs)
            model = keras.models.Model(inputs=inputs, outputs=outputs)

            output = running.process_whole_image(
                model, images,
                num_crops=num_crops,
                receptive_field=receptive_field,
                padding=None)


if __name__ == '__main__':
    test.main()
