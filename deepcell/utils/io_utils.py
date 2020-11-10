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
"""Utilities for reading/writing files"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
from skimage.io import imread
from skimage.external import tifffile as tiff
from skimage.external.tifffile import TiffFile
from tensorflow.keras import backend as K


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


def save_model_output(output,
                      output_dir,
                      feature_name='',
                      channel=None,
                      data_format=None):
    """Save model output as tiff images in the provided directory

    Args:
        output (numpy.array): Output of a model.
            Expects channel to have its own axis.
        output_dir (str): Directory to save the model output images.
        feature_name (str): Optional description to start each output image
            filename.
        channel (int): If given, only saves this channel.
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
