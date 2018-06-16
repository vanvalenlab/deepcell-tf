import pytest

from deepcell.utils.misc_utils import sorted_nicely


def test_sorted_nicely():
    # test image file sorting
    expected = ['test_image_001_dapi', 'test_image_002_dapi', 'test_image_003_dapi']
    unsorted = ['test_image_003_dapi', 'test_image_001_dapi', 'test_image_002_dapi']
    assert expected == sorted_nicely(unsorted)
    # test montage folder sorting
    expected = ['test_image_0_0', 'test_image_1_0', 'test_image_1_1']
    unsorted = ['test_image_1_1', 'test_image_0_0', 'test_image_1_0']
    assert expected == sorted_nicely(unsorted)

if __name__ == '__main__':
    pytest.main([__file__])
