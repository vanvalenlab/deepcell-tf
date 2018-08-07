"""
Tests for train_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import keras
from tensorflow.python.platform import test

from deepcell.utils.train_utils import rate_scheduler


class TrainUtilsTest(test.TestCase):
    def test_rate_scheduler(self):
        # if decay is small, learning rate should decrease as epochs increase
        rs = rate_scheduler(lr=.001, decay=.95)
        assert rs(1) > rs(2)
        # if decay is large, learning rate should increase as epochs increase
        rs = rate_scheduler(lr=.001, decay=1.05)
        assert rs(1) < rs(2)
        # if decay is 1, learning rate should not change
        rs = rate_scheduler(lr=.001, decay=1)
        assert rs(1) == rs(2)

if __name__ == '__main__':
    test.main()
