"""Tests for misc_utils"""

from tensorflow.python.platform import test

from deepcell.utils import misc_utils


class MiscUtilsTest(test.TestCase):
    def test_sorted_nicely(self):
        # test image file sorting
        expected = ['test_001_dapi', 'test_002_dapi', 'test_003_dapi']
        unsorted = ['test_003_dapi', 'test_001_dapi', 'test_002_dapi']
        self.assertListEqual(expected, misc_utils.sorted_nicely(unsorted))
        # test montage folder sorting
        expected = ['test_0_0', 'test_1_0', 'test_1_1']
        unsorted = ['test_1_1', 'test_0_0', 'test_1_0']
        self.assertListEqual(expected, misc_utils.sorted_nicely(unsorted))

    def test_get_sorted_keys(self):
        d = {'C1': 1, 'C3': 2, 'C2': 3}
        self.assertListEqual(misc_utils.get_sorted_keys(d), ['C1', 'C2', 'C3'])


if __name__ == '__main__':
    test.main()
