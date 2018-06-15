"""
running.py

Functions for running convolutional neural networks

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import warnings

import numpy as np
import tifffile as tiff
from tensorflow.python.keras import backend as K

from .utils.io_utils import get_images_from_directory
from .settings import CHANNELS_FIRST, CHANNELS_LAST

"""
Running convnets
"""

def run_model(image, model, win_x=30, win_y=30, split=True):
    # pad_width = ((0, 0), (0, 0), (win_x, win_x), (win_y, win_y))
    # image = np.pad(image, pad_width=pad_width , mode='constant', constant_values=0)
    channel_axis = 1 if CHANNELS_FIRST else -1
    x_axis = 2 if CHANNELS_FIRST else 1
    y_axis = 3 if CHANNELS_FIRST else 2

    evaluate_model = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output])

    n_features = model.layers[-1].output_shape[channel_axis]

    if split:
        warnings.warn('The split flag is deprecated and is designed to account '
                      'for a maximum tensor size.')

        image_size_x = image.shape[x_axis] // 2
        image_size_y = image.shape[y_axis] // 2

        if CHANNELS_FIRST:
            shape = (n_features, 2*image_size_x-win_x*2, 2*image_size_y-win_y*2)
        else:
            shape = (2*image_size_x-win_x*2, 2*image_size_y-win_y*2, n_features)

        model_output = np.zeros(shape, dtype=K.floatx())

        if CHANNELS_FIRST:
            img_0 = image[:, :, 0:image_size_x+win_x, 0:image_size_y+win_y]
            img_1 = image[:, :, 0:image_size_x+win_x, image_size_y-win_y:]
            img_2 = image[:, :, image_size_x-win_x:, 0:image_size_y+win_y]
            img_3 = image[:, :, image_size_x-win_x:, image_size_y-win_y:]

            model_output[:, 0:image_size_x-win_x, 0:image_size_y-win_y] = evaluate_model([img_0, 0])[0]
            model_output[:, 0:image_size_x-win_x, image_size_y-win_y:] = evaluate_model([img_1, 0])[0]
            model_output[:, image_size_x-win_x:, 0:image_size_y-win_y] = evaluate_model([img_2, 0])[0]
            model_output[:, image_size_x-win_x:, image_size_y-win_y:] = evaluate_model([img_3, 0])[0]
        else:
            img_0 = image[:, 0:image_size_x+win_x, 0:image_size_y+win_y, :]
            img_1 = image[:, 0:image_size_x+win_x, image_size_y-win_y:, :]
            img_2 = image[:, image_size_x-win_x:, 0:image_size_y+win_y, :]
            img_3 = image[:, image_size_x-win_x:, image_size_y-win_y:, :]

            model_output[0:image_size_x-win_x, 0:image_size_y-win_y, :] = evaluate_model([img_0, 0])[0]
            model_output[0:image_size_x-win_x, image_size_y-win_y:, :] = evaluate_model([img_1, 0])[0]
            model_output[image_size_x-win_x:, 0:image_size_y-win_y, :] = evaluate_model([img_2, 0])[0]
            model_output[image_size_x-win_x:, image_size_y-win_y:, :] = evaluate_model([img_3, 0])[0]

    else:
        model_output = evaluate_model([image, 0])[0]
        model_output = model_output[0, :, :, :]

    return model_output

def run_model_on_directory(data_location, channel_names, output_location, model,
                           win_x=30, win_y=30, split=True, save=True):

    channel_axis = 1 if CHANNELS_FIRST else -1
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
                feature = model_output[f, :, :] if CHANNELS_FIRST else model_output[:, :, f]
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

    if CHANNELS_LAST:
        input_shape = (input_shape[1], input_shape[2], input_shape[0])
        batch_shape = (1, input_shape[1], input_shape[2], input_shape[0])
    else:
        batch_shape = (1, input_shape[0], input_shape[1], input_shape[2])

    model = model_fn(input_shape=input_shape, n_features=n_features)

    for layer in model.layers:
        print(layer.name)

    channel_axis = 1 if CHANNELS_FIRST else -1
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
                feature = model_output[i, f, :, :] if CHANNELS_FIRST else model_output[i, :, :, f]
                cnnout_name = 'feature_{}_frame_{}.tif'.format(f, i)
                tiff.imsave(os.path.join(output_location, cnnout_name), feature)

    return model_output
