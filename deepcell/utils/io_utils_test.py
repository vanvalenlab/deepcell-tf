# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
"""Tests for io_utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.platform import test
from skimage.io import imsave

from deepcell.utils import io_utils


def _write_image(filepath, img_w=30, img_h=30):
    bias = np.random.rand(img_w, img_h, 1) * 64
    variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
    imarray = np.random.rand(img_w, img_h, 1) * variance + bias
    imsave(filepath, imarray[..., 0], check_contrast=False)


class TestIOUtils(test.TestCase):

    def test_get_image(self):
        image = 255 * np.random.random(size=(300, 300, 1)).astype('float32')
        temp_dir = self.get_temp_dir()
        # test tiff files
        test_img_path = os.path.join(temp_dir, 'phase.tif')
        imsave(test_img_path, image, check_contrast=False)
        test_img = io_utils.get_image(test_img_path)
        self.assertAllEqual(image, test_img)
        # test png files
        # pngs are integer only and don't have a channel axis.
        test_img_path = os.path.join(temp_dir, 'feature_0.png')
        image = image.astype('uint8')
        imsave(test_img_path, image, check_contrast=False)
        test_img = io_utils.get_image(test_img_path)
        test_img = np.expand_dims(test_img, axis=-1)
        self.assertAllClose(image, test_img)

    def test_save_model_output(self):
        temp_dir = self.get_temp_dir()
        batches = 1
        features = 3
        img_w, img_h, frames = 30, 30, 5

        # test channels_last
        K.set_image_data_format('channels_last')

        # test 2D output
        output = np.random.random((batches, img_w, img_h, features))
        io_utils.save_model_output(output, temp_dir, 'test', channel=None)
        # test saving only one channel
        io_utils.save_model_output(output, temp_dir, 'test', channel=1)

        # test 3D output
        output = np.random.random((batches, frames, img_w, img_h, features))
        io_utils.save_model_output(output, temp_dir, 'test', channel=None)
        # test saving only one channel
        io_utils.save_model_output(output, temp_dir, 'test', channel=1)

        # test channels_first 2D
        output = np.random.random((batches, features, img_w, img_h))
        io_utils.save_model_output(output, temp_dir, 'test', channel=None,
                                   data_format='channels_first')

        # test channels_first 3D
        output = np.random.random((batches, features, frames, img_w, img_h))
        io_utils.save_model_output(output, temp_dir, 'test', channel=None,
                                   data_format='channels_first')

        # test bad channel
        with self.assertRaises(ValueError):
            output = np.random.random((batches, features, img_w, img_h))
            io_utils.save_model_output(output, temp_dir, 'test', channel=-1)
            io_utils.save_model_output(output, temp_dir, 'test',
                                       channel=features + 1)

        # test no output directory
        with self.assertRaises(IOError):
            bad_dir = os.path.join(temp_dir, 'test')
            io_utils.save_model_output(output, bad_dir, 'test', channel=None)

if __name__ == '__main__':
    test.main()
