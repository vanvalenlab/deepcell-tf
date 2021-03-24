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
"""Tests for CellTracking Application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
