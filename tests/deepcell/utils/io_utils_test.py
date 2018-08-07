"""
Tests for io_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.platform import test
from skimage.external import tifffile as tiff

from deepcell.utils.io_utils import get_immediate_subdirs
from deepcell.utils.io_utils import get_image
from deepcell.utils.io_utils import nikon_getfiles
from deepcell.utils.io_utils import get_image_sizes
from deepcell.utils.io_utils import get_images_from_directory


def _write_image(filepath, img_w=30, img_h=30):
    bias = np.random.rand(img_w, img_h, 1) * 64
    variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
    imarray = np.random.rand(img_w, img_h, 1) * variance + bias
    if filepath.lower().endswith('tif') or filepath.lower().endswith('tiff'):
        tiff.imsave(filepath, imarray[:, :, 0])
    else:
        img = array_to_img(imarray, scale=False, data_format='channels_last')
        img.save(filepath)


class TestIOUtils(test.TestCase):

    def test_get_immediate_subdirs(self):
        dirs = []
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir)
        for x in range(2, -1, -1):  # iterate backwards to test sorting
            sub_dir = os.path.join(temp_dir, str(x))
            try:
                os.makedirs(sub_dir)
            except OSError as err:
                if err.errno != os.errno.EEXIST:
                    raise
            dirs.append(str(x))

        assert get_immediate_subdirs(temp_dir) == list(reversed(dirs))

    def test_get_image(self):
        temp_dir = self.get_temp_dir()
        # test tiff files
        test_img_path = os.path.join(temp_dir, 'phase.tif')
        _write_image(test_img_path, 300, 300)
        test_img = get_image(test_img_path)
        assert np.asarray(test_img).shape == (300, 300)
        # test png files
        test_img_path = os.path.join(temp_dir, 'feature_0.png')
        _write_image(test_img_path, 400, 400)
        test_img = get_image(test_img_path)
        assert np.asarray(test_img).shape == (400, 400)

    def test_nikon_getfiles(self):
        temp_dir = self.get_temp_dir()
        for filename in ('channel.tif', 'multi1.tif', 'multi2.tif'):
            _write_image(os.path.join(temp_dir, filename), 300, 300)

        images = nikon_getfiles(temp_dir, 'channel')
        assert images == ['channel.tif']
        multi_images = nikon_getfiles(temp_dir, 'multi')
        assert multi_images == ['multi1.tif', 'multi2.tif']
        no_images = nikon_getfiles(temp_dir, 'bad_channel_name')
        assert no_images == []

    def test_get_image_sizes(self):
        temp_dir = self.get_temp_dir()
        _write_image(os.path.join(temp_dir, 'image1.png'), 300, 300)
        _write_image(os.path.join(temp_dir, 'image2.png'), 300, 300)
        # test with single channel name
        size = get_image_sizes(temp_dir, ['image1'])
        assert size == (300, 300)
        # test with multiple channel names
        sizes = get_image_sizes(temp_dir, ['image1', 'image2'])
        assert sizes == (300, 300)

    def test_get_images_from_directory(self):
        temp_dir = self.get_temp_dir()
        _write_image(os.path.join(temp_dir, 'image.png'), 300, 300)
        # test channels_last
        K.set_image_data_format('channels_last')
        img = get_images_from_directory(temp_dir, ['image'])
        assert isinstance(img, list) and len(img) == 1
        assert img[0].shape == (1, 300, 300, 1)

        # test channels_last
        K.set_image_data_format('channels_first')
        img = get_images_from_directory(temp_dir, ['image'])
        assert isinstance(img, list) and len(img) == 1
        assert img[0].shape == (1, 1, 300, 300)

if __name__ == '__main__':
    test.main()
