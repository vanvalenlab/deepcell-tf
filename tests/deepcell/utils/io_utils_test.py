"""
Tests for io_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
from skimage.io import imread
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from deepcell.utils.io_utils import get_immediate_subdirs
from deepcell.utils.io_utils import get_image
from deepcell.utils.io_utils import nikon_getfiles
from deepcell.utils.io_utils import get_image_sizes
from deepcell.utils.io_utils import get_images_from_directory

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(TEST_DIR, 'resources')
TEST_IMG = imread(os.path.join(RES_DIR, 'phase.tif'))

class TestIOUtils(test.TestCase):

    def test_get_immediate_subdirs(self):
        dirs = []
        tmp_dir = os.path.join(RES_DIR, 'tmp')
        for x in range(2, -1, -1): # iterate backwards to test sorting
            sub_dir = os.path.join(tmp_dir, str(x))
            try:
                os.makedirs(sub_dir)
            except OSError as err:
                if err.errno != os.errno.EEXIST:
                    raise
            dirs.append(str(x))

        assert get_immediate_subdirs(tmp_dir) == list(reversed(dirs))
        shutil.rmtree(tmp_dir)
        # now tmp_dir is removed, so RES_DIR should have no subdirs
        assert get_immediate_subdirs(RES_DIR) == []

    def test_get_image(self):
        # test tiff files
        test_img_path = os.path.join(RES_DIR, 'phase.tif')
        test_img = get_image(test_img_path)
        assert np.asarray(test_img).shape == (300, 300)
        # test png files
        test_img_path = os.path.join(RES_DIR, 'feature_0.png')
        test_img = get_image(test_img_path)
        assert np.asarray(test_img).shape == (300, 300)

    def test_nikon_getfiles(self):
        images = nikon_getfiles(RES_DIR, 'phase')
        assert images == ['phase.tif']
        rotated_images = nikon_getfiles(RES_DIR, 'rotated')
        assert rotated_images == ['rotated_90.tif', 'rotated_180.tif', 'rotated_270.tif']
        no_images = nikon_getfiles(RES_DIR, 'bad_channel_name')
        assert no_images == []

    def test_get_image_sizes(self):
        # test with single channel name
        size = get_image_sizes(RES_DIR, ['phase'])
        assert size == (300, 300)
        # test with multiple channel names
        sizes = get_image_sizes(RES_DIR, ['phase', 'rotated'])
        assert sizes == (300, 300)

    def test_get_images_from_directory(self):
        # test channels_last
        K.set_image_data_format('channels_last')
        img = get_images_from_directory(RES_DIR, ['phase'])
        assert isinstance(img, list) and len(img) == 1
        assert img[0].shape == (1, 300, 300, 1)

        # test channels_last
        K.set_image_data_format('channels_first')
        img = get_images_from_directory(RES_DIR, ['phase'])
        assert isinstance(img, list) and len(img) == 1
        assert img[0].shape == (1, 1, 300, 300)

if __name__ == '__main__':
    test.main()
