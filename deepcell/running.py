"""
running.py

Functions for running convolutional neural networks

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import errno
import warnings

import numpy as np
from skimage.external import tifffile as tiff
from tensorflow.python.keras import backend as K

from deepcell.utils.io_utils import get_images_from_directory


def get_cropped_input_shape(images, num_crops=4):
    """Helper function to calculate the input_shape for models
    that will process cropped sub-images.
    # Arguments:
        images: numpy array of original data
        num_crops: number of slices for the x and y axis to create sub-images
    # Returns:
        input_shape: new input_shape for model to process sub-images.
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        row_axis = len(images.shape) - 2
        col_axis = len(images.shape) - 1
    else:
        channel_axis = len(images.shape) - 1
        row_axis = len(images.shape) - 3
        col_axis = len(images.shape) - 2

    channel_dim = images.shape[channel_axis]

    # Split the frames into quarters, as the full image size is too large
    crop_x = images.shape[row_axis] // num_crops
    crop_y = images.shape[col_axis] // num_crops

    if images.ndim == 5:
        input_shape = (images.shape[row_axis - 1], crop_x, crop_y, channel_dim)
    else:
        input_shape = (crop_x, crop_y, channel_dim)

    # switch to channels_first if necessary
    if channel_axis == 1:
        input_shape = (input_shape[-1], *input_shape[:-1])

    return input_shape


def process_whole_image(model, images, num_crops=4):
    """Slice images into num_crops * num_crops pieces, and use the model to
    process each small image.
    # Arguments:
        model: model that will process each small image
        images: numpy array that is too big for model.predict(images)
        num_crops: number of slices for the x and y axis to create sub-images
    # Returns:
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

    # Split the frames into quarters, as the full image size is too large
    crop_x = images.shape[row_axis] // num_crops
    crop_y = images.shape[col_axis] // num_crops

    # instantiate matrix for model output
    model_output_shape = tuple(list(model.layers[-1].output_shape)[1:])
    output = np.zeros((images.shape[0], *model_output_shape))

    if any([images.shape[i] != output.shape[i] for i in (row_axis, col_axis)]):
        raise ValueError('Expected images and model output to have same shape,'
                         ' Got: {} and {}, respectively'.format(
                             images.shape, output.shape))

    expected_input_shape = get_cropped_input_shape(images, num_crops)
    if expected_input_shape != model.input_shape:
        raise ValueError('Expected model.input_shape to be {}. Got {}.  Use '
                         '`get_new_input_shape()` to recreate your model with '
                         'the proper input_shape'.format(
                             expected_input_shape, model.input_shape))

    # Slice the images into smaller sub-images
    y_split = np.split(images, num_crops, axis=col_axis)
    images_split = [np.split(s, num_crops, axis=row_axis) for s in y_split]

    for i in range(num_crops):
        for j in range(num_crops):
            predicted = model.predict(images_split[j][i])
            a, b = i * crop_x, (i + 1) * crop_x
            c, d = j * crop_y, (j + 1) * crop_y
            if images.ndim == 5:
                if channel_axis == 1:
                    output[:, :, a:b, c:d, :] = predicted
                else:
                    output[:, :, :, a:b, c:d] = predicted
            else:
                if channel_axis == 1:
                    output[:, a:b, c:d, :] = predicted
                else:
                    output[:, :, a:b, c:d] = predicted
    return output


def save_model_output(output, output_dir, feature_name='', channel=None):
    """Save model output as tiff images in the provided directory
    # Arguments:
        output: output of model. Expects channel to have its own axis
        output_dir: directory to save the model output images
        feature_name: optional description to start each output image filename
        channel: if given,only saves this channel
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    channel_axis = 1 if is_channels_first else -1
    z_axis = 2 if is_channels_first else 1

    if 0 > channel > output.shape[channel_axis]:
        raise ValueError('`channel` must be in the range of the output '
                         'channels. Got ', channel)

    for b in range(output.shape[0]):
        # If multiple batches of results, create a numbered subdirectory
        batch_dir = str(b) if output.shape[0] > 1 else ''

        try:
            os.makedirs(os.path.join(output_dir, batch_dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        # If 2D, convert to 3D with only one z-axis
        if len(output.shape) == 4:
            output = np.expand_dims(output, axis=z_axis)

        for f in range(output.shape[z_axis]):
            for c in range(output.shape[channel_axis]):
                # if only saving one channel, skip the non-equal channels
                if channel is not None and channel != c:
                    continue

                if is_channels_first:
                    feature = output[b, c, f, :, :]
                else:
                    feature = output[b, f, :, :, c]

                zpad = max(3, len(str(output.shape[z_axis])))
                cnnout_name = 'feature_{}_frame_{}.tif'.format(c, str(f).zfill(zpad))
                if feature_name:
                    cnnout_name = '{}_{}'.format(feature_name, cnnout_name)

                out_file_path = os.path.join(output_dir, batch_dir, cnnout_name)
                tiff.imsave(out_file_path, feature.astype('int32'))
        print('Saved {} frames to {}'.format(output.shape[1], output_dir))


def run_model(image, model, win_x=30, win_y=30, split=True):
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
