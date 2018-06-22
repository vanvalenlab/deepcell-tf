import os

import pytest
from skimage.io import imread

from deepcell.utils.plot_utils import cf

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(TEST_DIR, 'resources')
TEST_IMG = imread(os.path.join(RES_DIR, 'phase.tif'))

def test_cf():
    # values are hard-coded for test image
    shape = TEST_IMG.shape
    # test coordinates outside of test_img dimensions
    assert cf(shape[0] + 1, shape[1] + 1, TEST_IMG) == 'x=301.0000, y=301.0000'
    assert cf(-1 * shape[0], -1 * shape[1], TEST_IMG) == 'x=-300.0000, y=-300.0000'
    # test coordinates inside test_img dimensions
    assert cf(shape[0] / 2, shape[1] / 2, TEST_IMG) == 'x=150.0000, y=150.0000, z=7771.0000'

if __name__ == '__main__':
    pytest.main([__file__])
