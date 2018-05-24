"""
dc_running_functions.py

Functions for running convolutional neural networks

@author: David Van Valen
"""

from __future__ import print_function
from __future__ import division

import os

import numpy as np
from tensorflow.python.keras import backend as K

import tifffile as tiff
from .dc_helper_functions import *

"""
Running convnets
"""

def run_model(image, model, win_x=30, win_y=30, std=False, split=True, process=True):
    # image = np.pad(image, pad_width = ((0,0), (0,0), (win_x, win_x),(win_y,win_y)), mode = 'constant', constant_values = 0)
    image_data_format = K.image_data_format()
    channels_axis = 1 if image_data_format == 'channels_first' else -1
    x_index = 2 if image_data_format == 'channels_first' else 1
    y_index = 3 if image_data_format == 'channels_first' else 2

    if process:
        for j in range(image.shape[channels_axis]):
            if image_data_format == 'channels_first':
                image[0, j, :, :] = process_image(image[0, j, :, :], win_x, win_y, std)
            else:
                image[0, :, :, j] = process_image(image[0, :, :, j], win_x, win_y, std)

    if split:
        image_size_x = image.shape[x_index] // 2
        image_size_y = image.shape[y_index] // 2
    else:
        image_size_x = image.shape[x_index]
        image_size_y = image.shape[y_index]

    evaluate_model = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output])

    n_features = model.layers[-1].output_shape[channels_axis]

    if split:
        if image_data_format == 'channels_first':
            shape = (n_features, 2*image_size_x-win_x*2, 2*image_size_y-win_y*2)
        else:
            shape = (2*image_size_x-win_x*2, 2*image_size_y-win_y*2, n_features)

        model_output = np.zeros(shape, dtype='float32')

        # TODO: what if image_data_format == 'channels_last'?
        if image_data_format == 'channels_first':
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
                           win_x=30, win_y=30, std=False, split=True, process=True, save=True):
    image_data_format = K.image_data_format()
    channels_axis = 1 if image_data_format == 'channels_first' else -1

    n_features = model.layers[-1].output_shape[channels_axis]

    image_list = get_images_from_directory(data_location, channel_names)
    processed_image_list = []

    for image in image_list:
        print("Processing image " + str(counter + 1) + " of " + str(len(image_list)))
        processed_image = run_model(image, model, win_x=win_x, win_y=win_y,
                                    std=std, split=split, process=process)
        processed_image_list += [processed_image]

        # Save images
        if save:
            for feat in range(n_features):
                cnnout_name = os.path.join(output_location, 'feature_' + str(feat) +"_frame_"+ str(counter) + r'.tif')
                tiff.imsave(cnnout_name, feature)
        counter += 1
                if image_data_format == 'channels_first':
                    feature = processed_image[feat, :, :]
                else:
                    feature = processed_image[:, :, feat]

    return processed_image_list

def run_models_on_directory(data_location, channel_names, output_location, model_fn, list_of_weights, n_features = 3, image_size_x = 1080, image_size_y = 1280, win_x = 30, win_y = 30, std = False, split = True, process = True, save = True):
    image_data_format = K.image_data_format()
    channels_axis = 1 if image_data_format == 'channels_first' else -1

    if split:
        input_shape = (len(channel_names), image_size_x // 2 + win_x, image_size_y // 2 + win_y)
    else:
        input_shape = (len(channel_names), image_size_x, image_size_y)

    if image_data_format == 'channels_last':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

    batch_shape = (1, input_shape[0], input_shape[1], input_shape[2])
    model = model_fn(batch_shape=batch_shape, n_features=n_features)

    for layer in model.layers:
        print(layer.name)
    n_features = model.layers[-1].output_shape[channels_axis]

    model_outputs = []
    for weights_path in list_of_weights:
        model.load_weights(weights_path)
        processed_image_list = run_model_on_directory(data_location, channel_names,
                                                      output_location, model,
                                                      win_x=win_x, win_y=win_y,
                                                      save=False, std=std, split=split,
                                                      process=process)

        model_outputs += [np.stack(processed_image_list, axis=0)]

    # Average all images
    model_output = np.stack(model_outputs, axis=0)
    model_output = np.mean(model_output, axis=0)

    # Save images
    if save:
        for img in range(model_output.shape[0]):
            for feat in range(n_features):
                cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(img) + r'.tif')
                tiff.imsave(cnnout_name, feature)
                if image_data_format == 'channels_first':
                    feature = model_output[img, feat, :, :]
                else:
                    feature = model_output[img, :, :, feat]

    return model_output
