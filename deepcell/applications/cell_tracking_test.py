"""Tests for CellTracking Application"""


from tensorflow.python.platform import test
import numpy as np
import skimage as sk

from deepcell.model_zoo.tracking import GNNTrackingModel
from deepcell.applications import CellTracking


def _get_dummy_tracking_data(length=128, frames=3,
                             data_format='channels_last'):
    """Borrowed from deepcell-tracking: https://bit.ly/37MFuNQ"""
    if data_format == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 0

    x, y = [], []
    while len(x) < frames:
        _x = sk.data.binary_blobs(length=length, n_dim=2)
        _y = sk.measure.label(_x)
        if len(np.unique(_y)) > 3:
            x.append(_x)
            y.append(_y)

    x = np.stack(x, axis=0)  # expand to 3D
    y = np.stack(y, axis=0)  # expand to 3D

    x = np.expand_dims(x, axis=channel_axis)
    y = np.expand_dims(y, axis=channel_axis)

    return x.astype('float32'), y.astype('int32')


class TestCellTracking(test.TestCase):

    def test_cell_tracking_app(self):
        with self.cached_session():
            # Instantiate model
            tm = GNNTrackingModel()

            # Test instantiation
            app = CellTracking(model=tm.inference_model,
                               neighborhood_encoder=tm.neighborhood_encoder)

            # test output shape
            shape = app.model.output_shape
            self.assertIsInstance(shape, tuple)
            self.assertEqual(shape[-1], 3)

            # test predict
            x, y = _get_dummy_tracking_data(128, frames=3)
            tracked = app.predict(x, y)
            self.assertEqual(tracked['X'].shape, tracked['y_tracked'].shape)
