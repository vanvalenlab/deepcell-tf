# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from deepcell.utils.data_utils import trim_padding


def get_cropped_input_shape(images,
                            num_crops=4,
                            receptive_field=61,
                            data_format=None):
    """Calculate the input_shape for models to process cropped sub-images.

    Args:
        images (numpy.array): numpy array of original data
        num_crops (int): number of slices for the x and y axis
            to create sub-images
        receptive_field (int): the receptive field of the neural network.
        data_format (str): "channels_first" or "channels_last"

    Returns:
        tuple: new ``input_shape`` for model to process sub-images.
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
        input_shape = tuple([input_shape[-1]] + list(input_shape[:-1]))

    return input_shape


def get_padding_layers(model):
    """Get all names of padding layers in a model

    Args:
        model (tensorflow.keras.Model): Keras model

    Returns:
        list: list of names of padding layers inside model
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
        model (tensorflow.keras.Model): model that will process each small image
        images (numpy.array): numpy array that is too big for model.predict
        num_crops (int): number of slices for the x and y axis
            to create sub-images
        receptive_field (int): receptive field used by model,
            required to pad images
        padding (str): type of padding for input images,
            one of {'reflect', 'zero'}.

    Returns:
        numpy.array: model outputs for each sub-image

    Raises:
        ValueError: invalid padding value
        ValueError: model input shape is different than expected_input_shape
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
        output = np.zeros(tuple([images.shape[0], model_output_shape[0]] +
                                list(images.shape[2:])))
    else:
        output = np.zeros(tuple(list(images.shape[0:-1]) +
                                [model_output_shape[-1]]))

    expected_input_shape = get_cropped_input_shape(
        images, num_crops, receptive_field)

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
