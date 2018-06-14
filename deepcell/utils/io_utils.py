"""
io_utils.py

utility functions for reading/writing files

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
from skimage.io import imread
from tifffile import TiffFile
from tensorflow.python.keras import backend as K

from .misc_utils import sorted_nicely

def get_immediate_subdirs(directory):
    """
    Get all DIRECTORIES that are immediate children of a given directory
    # Arguments
        dir: a filepath to a directory
    # Returns:
        a sorted list of child directories of given dir.  Will not return files.
    """
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def get_image(file_name):
    """
    Read image from file and load into numpy array
    """
    if os.path.splitext(file_name.lower())[-1] == '.tif':
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))

def nikon_getfiles(direc_name, channel_name):
    """
    Return all image filenames in direc_name with
    channel_name in the filename
    """
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if channel_name in i]
    imgfiles = sorted_nicely(imgfiles)
    return imgfiles

def get_image_sizes(data_location, channel_names):
    """Get the first image inside the data_location and return its shape"""
    img_list_channels = []
    for channel in channel_names:
        img_list_channels.append(nikon_getfiles(data_location, channel))
    img_temp = np.asarray(get_image(os.path.join(data_location, img_list_channels[0][0])))
    return img_temp.shape

def get_images_from_directory(data_location, channel_names):
    """
    Read all images from directory with channel_name in the filename
    Return them in a numpy array
    """
    data_format = K.image_data_format()
    img_list_channels = []
    for channel in channel_names:
        img_list_channels.append(nikon_getfiles(data_location, channel))

    img_temp = np.asarray(get_image(os.path.join(data_location, img_list_channels[0][0])))

    n_channels = len(channel_names)
    all_images = []

    for stack_iteration in range(len(img_list_channels[0])):
        if data_format == 'channels_first':
            shape = (1, n_channels, img_temp.shape[0], img_temp.shape[1])
        else:
            shape = (1, img_temp.shape[0], img_temp.shape[1], n_channels)

        all_channels = np.zeros(shape, dtype=K.floatx())

        for j in range(n_channels):
            img_path = os.path.join(data_location, img_list_channels[j][stack_iteration])
            channel_img = get_image(img_path)
            if data_format == 'channels_first':
                all_channels[0, j, :, :] = channel_img
            else:
                all_channels[0, :, :, j] = channel_img

        all_images.append(all_channels)

    return all_images
