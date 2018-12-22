# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Functions for running convolutional neural networks"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import warnings

import numpy as np
from skimage.external import tifffile as tiff
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

from deepcell.utils.data_utils import trim_padding
from deepcell.utils.io_utils import get_images_from_directory


def get_cropped_input_shape(images, num_crops=4, receptive_field=61, data_format=None):
    """Helper function to calculate the input_shape for models
    that will process cropped sub-images.

    Args:
        images: numpy array of original data
        num_crops: number of slices for the x and y axis to create sub-images

    Returns:
        input_shape: new input_shape for model to process sub-images.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        channel_axis = 1
        row_axis = len(images.shape) - 2
        col_axis = len(images.shape) - 1
    else:
        channel_axis = len(images.shape) - 1
        row_axis = len(images.shape) - 3
        col_axis = len(images.shape) - 2

    channel_dim = images.shape[channel_axis]

    # Split the frames into quarters, as the full image size is too large
    crop_x = images.shape[row_axis] // num_crops + (receptive_field - 1)
    crop_y = images.shape[col_axis] // num_crops + (receptive_field - 1)

    if images.ndim == 5:
        input_shape = (images.shape[row_axis - 1], crop_x, crop_y, channel_dim)
    else:
        input_shape = (crop_x, crop_y, channel_dim)

    # switch to channels_first if necessary
    if channel_axis == 1:
        input_shape = (input_shape[-1], *input_shape[:-1])

    return input_shape


def get_padding_layers(model):
    """Get all names of padding layers in the model

    Args:
        model: Keras model

    Returns:
        padding_layers: list of names of padding layers inside model
    """
    padding_layers = []
    for layer in model.layers:
        if 'padding' in layer.name:
            padding_layers.append(layer.name)
        elif isinstance(layer, Model):
            padding_layers.extend(get_padding_layers(layer))
    return padding_layers


def process_whole_image(model, images, num_crops=4, receptive_field=61, padding=None):
    """Slice images into num_crops * num_crops pieces, and use the model to
    process each small image.

    Args:
        model: model that will process each small image
        images: numpy array that is too big for model.predict(images)
        num_crops: number of slices for the x and y axis to create sub-images
        receptive_field: receptive field used by model, required to pad images
        padding: type of padding for input images, one of {'reflect', 'zero'}

    Returns:
        model_output: numpy array containing model outputs for each sub-image
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        row_axis = len(images.shape) - 2
        col_axis = len(images.shape) - 1
    else:
        channel_axis = len(images.shape) - 1
        row_axis = len(images.shape) - 3
        col_axis = len(images.shape) - 2

    if not padding:
        padding_layers = get_padding_layers(model)
        if padding_layers:
            padding = 'reflect' if 'reflect' in padding_layers[0] else 'zero'

    if str(padding).lower() not in {'reflect', 'zero'}:
        raise ValueError('Expected `padding_mode` to be either `zero` or '
                         '`reflect`.  Got ', padding)

    # Split the frames into quarters, as the full image size is too large
    crop_x = images.shape[row_axis] // num_crops
    crop_y = images.shape[col_axis] // num_crops

    # Set up receptive field window for padding
    win_x, win_y = (receptive_field - 1) // 2, (receptive_field - 1) // 2

    # instantiate matrix for model output
    model_output_shape = tuple(list(model.layers[-1].output_shape)[1:])
    if channel_axis == 1:
        output = np.zeros((images.shape[0], model_output_shape[1], *images.shape[2:]))
    else:
        output = np.zeros((*images.shape[0:-1], model_output_shape[-1]))

    expected_input_shape = get_cropped_input_shape(images, num_crops, receptive_field)
    if expected_input_shape != model.input_shape[1:]:
        raise ValueError('Expected model.input_shape to be {}. Got {}.  Use '
                         '`get_cropped_input_shape()` to recreate your model '
                         ' with the proper input_shape'.format(
                             expected_input_shape, model.input_shape[1:]))

    # pad the images only in the x and y axes
    pad_width = []
    for i in range(len(images.shape)):
        if i == row_axis:
            pad_width.append((win_x, win_x))
        elif i == col_axis:
            pad_width.append((win_y, win_y))
        else:
            pad_width.append((0, 0))

    if str(padding).lower() == 'reflect':
        padded_images = np.pad(images, pad_width, mode='reflect')
    else:
        padded_images = np.pad(images, pad_width, mode='constant', constant_values=0)

    for i in range(num_crops):
        for j in range(num_crops):
            e, f = i * crop_x, (i + 1) * crop_x + 2 * win_x
            g, h = j * crop_y, (j + 1) * crop_y + 2 * win_y

            if images.ndim == 5:
                if channel_axis == 1:
                    predicted = model.predict(padded_images[:, :, :, e:f, g:h])
                else:
                    predicted = model.predict(padded_images[:, :, e:f, g:h, :])
            else:
                if channel_axis == 1:
                    predicted = model.predict(padded_images[:, :, e:f, g:h])
                else:
                    predicted = model.predict(padded_images[:, e:f, g:h, :])

            # if using skip_connections, get the final model output
            if isinstance(predicted, list):
                predicted = predicted[-1]

            # if the model uses padding, trim the output images to proper shape
            # if model does not use padding, images should already be correct
            if padding:
                predicted = trim_padding(predicted, win_x, win_y)

            a, b = i * crop_x, (i + 1) * crop_x
            c, d = j * crop_y, (j + 1) * crop_y

            if images.ndim == 5:
                if channel_axis == 1:
                    output[:, :, :, a:b, c:d] = predicted
                else:
                    output[:, :, a:b, c:d, :] = predicted
            else:
                if channel_axis == 1:
                    output[:, :, a:b, c:d] = predicted
                else:
                    output[:, a:b, c:d, :] = predicted

    return output


def run_model(image, model, win_x=30, win_y=30, split=True):
    """Runs the chosen model.

    Args:
        model: model that will process each small image
        image: numpy array that is too big for model.predict(images)

    Returns:
        model_output: numpy array containing model outputs for each sub-image
    """
    # pad_width = ((0, 0), (0, 0), (win_x, win_x), (win_y, win_y))
    # image = np.pad(image, pad_width=pad_width , mode='constant', constant_values=0)
    is_channels_first = K.image_data_format() == 'channels_first'
    channel_axis = 1 if is_channels_first else -1
    x_axis = 2 if is_channels_first else 1
    y_axis = 3 if is_channels_first else 2

    n_features = model.layers[-1].output_shape[channel_axis]

    if split:
        warnings.warn('The split flag is deprecated and is designed to account '
                      'for a maximum tensor size.')

        image_size_x = image.shape[x_axis] // 2
        image_size_y = image.shape[y_axis] // 2

        if is_channels_first:
            shape = (n_features, 2 * image_size_x - win_x * 2, 2 * image_size_y - win_y * 2)
        else:
            shape = (2 * image_size_x - win_x * 2, 2 * image_size_y - win_y * 2, n_features)

        model_output = np.zeros(shape, dtype=K.floatx())

        if is_channels_first:
            img_0 = image[:, :, 0:image_size_x + win_x, 0:image_size_y + win_y]
            img_1 = image[:, :, 0:image_size_x + win_x, image_size_y - win_y:]
            img_2 = image[:, :, image_size_x - win_x:, 0:image_size_y + win_y]
            img_3 = image[:, :, image_size_x - win_x:, image_size_y - win_y:]

            model_output[:, 0:image_size_x - win_x, 0:image_size_y - win_y] = model.predict(img_0)
            model_output[:, 0:image_size_x - win_x, image_size_y - win_y:] = model.predict(img_1)
            model_output[:, image_size_x - win_x:, 0:image_size_y - win_y] = model.predict(img_2)
            model_output[:, image_size_x - win_x:, image_size_y - win_y:] = model.predict(img_3)
        else:
            img_0 = image[:, 0:image_size_x + win_x, 0:image_size_y + win_y, :]
            img_1 = image[:, 0:image_size_x + win_x, image_size_y - win_y:, :]
            img_2 = image[:, image_size_x - win_x:, 0:image_size_y + win_y, :]
            img_3 = image[:, image_size_x - win_x:, image_size_y - win_y:, :]

            model_output[0:image_size_x - win_x, 0:image_size_y - win_y, :] = model.predict(img_0)
            model_output[0:image_size_x - win_x, image_size_y - win_y:, :] = model.predict(img_1)
            model_output[image_size_x - win_x:, 0:image_size_y - win_y, :] = model.predict(img_2)
            model_output[image_size_x - win_x:, image_size_y - win_y:, :] = model.predict(img_3)

    else:
        model_output = model.predict(image)
        model_output = model_output[0, :, :, :]

    return model_output


def run_model_on_directory(data_location, channel_names, output_location, model,
                           win_x=30, win_y=30, split=True, save=True):

    is_channels_first = K.image_data_format() == 'channels_first'
    channel_axis = 1 if is_channels_first else -1
    n_features = model.layers[-1].output_shape[channel_axis]

    image_list = get_images_from_directory(data_location, channel_names)
    model_outputs = []

    for i, image in enumerate(image_list):
        print('Processing image {} of {}'.format(i + 1, len(image_list)))
        model_output = run_model(image, model, win_x=win_x, win_y=win_y, split=split)
        model_outputs.append(model_output)

        # Save images
        if save:
            for f in range(n_features):
                feature = model_output[f, :, :] if is_channels_first else model_output[:, :, f]
                cnnout_name = 'feature_{}_frame_{}.tif'.format(f, str(i).zfill(3))
                tiff.imsave(os.path.join(output_location, cnnout_name), feature)

    return model_outputs


def run_models_on_directory(data_location, channel_names, output_location, model_fn,
                            list_of_weights, n_features=3, win_x=30, win_y=30,
                            image_size_x=1080, image_size_y=1280, save=True, split=True):
    if split:
        input_shape = (len(channel_names), image_size_x // 2 + win_x, image_size_y // 2 + win_y)
    else:
        input_shape = (len(channel_names), image_size_x, image_size_y)

    is_channels_first = K.image_data_format() == 'channels_first'
    if not is_channels_first:
        input_shape = (input_shape[1], input_shape[2], input_shape[0])
        batch_shape = (1, input_shape[1], input_shape[2], input_shape[0])
    else:
        batch_shape = (1, input_shape[0], input_shape[1], input_shape[2])

    model = model_fn(input_shape=input_shape, n_features=n_features)

    for layer in model.layers:
        print(layer.name)

    channel_axis = 1 if is_channels_first else -1
    n_features = model.layers[-1].output_shape[channel_axis]

    model_outputs = []
    for weights_path in list_of_weights:
        model.load_weights(weights_path)
        processed_image_list = run_model_on_directory(
            data_location, channel_names, output_location, model,
            win_x=win_x, win_y=win_y, save=False, split=split)

        model_outputs.append(np.stack(processed_image_list, axis=0))

    # Average all images
    model_output = np.stack(model_outputs, axis=0)
    model_output = np.mean(model_output, axis=0)

    # Save images
    if save:
        for i in range(model_output.shape[0]):
            for f in range(n_features):
                if is_channels_first:
                    feature = model_output[i, f, :, :]
                else:
                    feature = model_output[i, :, :, f]
                cnnout_name = 'feature_{}_frame_{}.tif'.format(f, i)
                tiff.imsave(os.path.join(output_location, cnnout_name), feature)

    return model_output
