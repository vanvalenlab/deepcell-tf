"""Tests for train_utils"""

import os

from tensorflow.keras import callbacks
from tensorflow.python.platform import test

from deepcell.utils import train_utils


class TrainUtilsTest(test.TestCase):

    def test_get_callbacks(self):
        temp_dir = self.get_temp_dir()
        model_path = os.path.join(temp_dir, 'test_model.h5')
        log_dir = os.path.join(temp_dir, 'logs')

        # test minimum set of callbacks:
        cbs = train_utils.get_callbacks(model_path)
        assert len(cbs) == 1
        assert isinstance(cbs[0], callbacks.ModelCheckpoint)

        # test all options
        cbs = train_utils.get_callbacks(model_path,
                                        reduce_lr_on_plateau=True,
                                        lr_sched=train_utils.rate_scheduler(),
                                        tensorboard_log_dir=log_dir)
        assert len(cbs) == 4
        assert isinstance(cbs[1], callbacks.LearningRateScheduler)
        assert isinstance(cbs[2], callbacks.ReduceLROnPlateau)
        assert isinstance(cbs[3], callbacks.TensorBoard)

    def test_rate_scheduler(self):
        # if decay is small, learning rate should decrease as epochs increase
        rs = train_utils.rate_scheduler(lr=.001, decay=.95)
        self.assertGreater(rs(1), rs(2))
        # if decay is large, learning rate should increase as epochs increase
        rs = train_utils.rate_scheduler(lr=.001, decay=1.05)
        self.assertLess(rs(1), rs(2))
        # if decay is 1, learning rate should not change
        rs = train_utils.rate_scheduler(lr=.001, decay=1)
        self.assertEqual(rs(1), rs(2))

if __name__ == '__main__':
    test.main()
