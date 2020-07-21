# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""Utilities for reading/writing files"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
from skimage.io import imread
from skimage.external import tifffile as tiff
from skimage.external.tifffile import TiffFile
from tensorflow.python.keras import backend as K

from deepcell.utils.misc_utils import sorted_nicely


def get_immediate_subdirs(directory):
    """Get all DIRECTORIES that are immediate children of a given directory.

    Args:
        directory (str): a filepath to a directory

    Returns:
        list: a sorted list of child directories of given dir.
    """
    exists = lambda x: os.path.isdir(os.path.join(directory, x))
    return sorted([d for d in os.listdir(directory) if exists(d)])


def count_image_files(directory, montage_mode=False):
    """Counts all image files inside the directory.
    If montage_mode, counts 1 level deep and returns the minimum count.
    Else, counts all child images of directory.

    Args:
        directory (str): directory to look for child image files
        montage_mode (bool): whether ot not to look in subdirs of directory

    Returns:
        int: the number of image files in the directory
    """
    def count_images(d):
        valid_extensions = {'.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp'}
        count = 0
        for f in os.listdir(directory):
            _, ext = os.path.splitext(f.lower())
            if ext in valid_extensions:
                count += 1
        return count

    if not montage_mode:
        return count_images(directory)
    return min([count_images(d) for d in get_immediate_subdirs(directory)])


def get_image(file_name):
    """Read image from file and returns it as a tensor

    Args:
        file_name (str): path to image file

    Returns:
        numpy.array: numpy array of image data
    """
    ext = os.path.splitext(file_name.lower())[-1]
    if ext == '.tif' or ext == '.tiff':
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))


def nikon_getfiles(direc_name, channel_name):
    """Return a sorted list of files inside direc_name
    with channel_name in the filename.

    Args:
        direc_name (str): directory to find image files
        channel_name (str): wildcard filter for filenames

    Returns:
        list: sorted list of files inside direc_name.
    """
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if channel_name in i]
    imgfiles = sorted_nicely(imgfiles)
    return imgfiles


def get_image_sizes(data_location, channel_names):
    """Get the first image inside the data_location and return its shape

    Args:
        data_location (str): path to image data
        channel_names (str[]): list of wildcards to filter filenames

    Returns:
        int: size of random image inside the data_location.
    """
    img_list_channels = []
    for channel in channel_names:
        img_list_channels.append(nikon_getfiles(data_location, channel))
    img_temp = np.asarray(get_image(os.path.join(data_location, img_list_channels[0][0])))
    return img_temp.shape


def get_images_from_directory(data_location, channel_names):
    """Read all images from directory with channel_name in the filename

    Args:
        data_location (str): folder containing image files
        channel_names (str[]): list of wildcards to select filenames

    Returns:
        numpy.array: numpy array of each image in the directory
    """
    data_format = K.image_data_format()
    img_list_channels = []
    for channel in channel_names:
        img_list_channels.append(nikon_getfiles(data_location, channel))

    img_temp = np.asarray(get_image(os.path.join(data_location, img_list_channels[0][0])))

    n_channels = len(channel_names)
    all_images = []

    for stack_iteration in range(len(img_list_channels[0])):

        if (data_format == 'channels_first') and (img_temp.ndim > 2):
            shape = (1, n_channels, img_temp.shape[0], img_temp.shape[1], img_temp.shape[2])
        elif data_format == 'channels_first':
            shape = (1, n_channels, img_temp.shape[0], img_temp.shape[1])
        elif img_temp.ndim > 2:
            shape = (1, img_temp.shape[0], img_temp.shape[1], img_temp.shape[2], n_channels)
        else:
            shape = (1, img_temp.shape[0], img_temp.shape[1], n_channels)

        all_channels = np.zeros(shape, dtype=K.floatx())

        for j in range(n_channels):
            img_path = os.path.join(data_location, img_list_channels[j][stack_iteration])
            channel_img = get_image(img_path)
            if data_format == 'channels_first':
                all_channels[0, j, ...] = channel_img
            else:
                all_channels[0, ..., j] = channel_img

        all_images.append(all_channels)

    return all_images


def save_model_output(output,
                      output_dir,
                      feature_name='',
                      channel=None,
                      data_format=None):
    """Save model output as tiff images in the provided directory

    Args:
        output (numpy.array): output of model. Expects channel to have its own axis
        output_dir (str): directory to save the model output images
        feature_name (str): optional description to start each output image filename
        channel (int): if given, only saves this channel
    """
    if data_format is None:
        data_format = K.image_data_format()
    channel_axis = 1 if data_format == 'channels_first' else -1
    z_axis = 2 if data_format == 'channels_first' else 1

    if channel is not None and not 0 < channel < output.shape[channel_axis]:
        raise ValueError('`channel` must be in the range of the output '
                         'channels. Got ', channel)

    if not os.path.isdir(output_dir):
        raise IOError('{} is not a valid output_dir'.format(
            output_dir))

    for b in range(output.shape[0]):
        # If multiple batches of results, create a numbered subdirectory
        batch_dir = str(b) if output.shape[0] > 1 else ''

        # If 2D, convert to 3D with only one z-axis
        if len(output.shape) == 4:
            output = np.expand_dims(output, axis=z_axis)

        for f in range(output.shape[z_axis]):
            for c in range(output.shape[channel_axis]):
                # if only saving one channel, skip the non-equal channels
                if channel is not None and channel != c:
                    continue

                if data_format == 'channels_first':
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
