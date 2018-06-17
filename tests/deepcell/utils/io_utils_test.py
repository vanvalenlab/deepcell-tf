import os
import shutil

import numpy as np
import pytest
from skimage.io import imread

from deepcell.utils.io_utils import get_immediate_subdirs
from deepcell.utils.io_utils import get_image
from deepcell.utils.io_utils import nikon_getfiles
from deepcell.utils.io_utils import get_image_sizes
from deepcell.utils.io_utils import get_images_from_directory

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(TEST_DIR, 'deepcell', 'resources')
TEST_IMG = imread(os.path.join(RES_DIR, 'phase.tif'))

def test_get_immediate_subdirs():
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

def test_get_image():
    test_img_path = os.path.join(RES_DIR, 'phase.tif')
    test_img = get_image(test_img_path)
    assert np.asarray(test_img).shape == (300, 300)

def test_nikon_getfiles():
    images = nikon_getfiles(RES_DIR, 'phase')
    assert images == ['phase.tif']
    rotated_images = nikon_getfiles(RES_DIR, 'rotated')
    assert rotated_images == ['rotated_90.tif', 'rotated_180.tif', 'rotated_270.tif']
    no_images = nikon_getfiles(RES_DIR, 'bad_channel_name')
    assert no_images == []

def test_get_image_sizes():
    # test with single channel name
    size = get_image_sizes(RES_DIR, ['phase'])
    assert size == (300, 300)
    # test with multiple channel names
    sizes = get_image_sizes(RES_DIR, ['phase', 'rotated'])
    assert sizes == (300, 300)

def test_get_images_from_directory():
    img = get_images_from_directory(RES_DIR, ['phase'])
    assert isinstance(img, list) and len(img) == 1
    assert img[0].shape == (1, 300, 300, 1)

if __name__ == '__main__':
    pytest.main([__file__])
