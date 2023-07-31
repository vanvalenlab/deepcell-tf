# Copyright 2016-2023 The Van Valen Lab at the California Institute of
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
import os

import numpy as np
import pandas as pd

from deepcell_tracking.trk_io import save_trks

from deepcell.datasets import SegmentationDataset, TrackingDataset


class TestSegmentationDataset:
    def test_no_meta(self, tmpdir, mocker):
        def mock_get_data(self):
            return str(tmpdir)
        mocker.patch('deepcell.datasets.SegmentationDataset._get_data', mock_get_data)

        dataset = SegmentationDataset('', '')

        # Create test data to load
        shape = (1, 10, 10, 1)
        split = 'test'
        np.savez_compressed(os.path.join(str(tmpdir), f'{split}.npz'), X=np.zeros(shape), y=np.zeros(shape))

        X, y, meta = dataset.load_data(split=split)
        assert X.shape == shape
        assert y.shape == shape
        assert meta == None

    def test_meta(self, tmpdir, mocker):
        def mock_get_data(self):
            return str(tmpdir)
        mocker.patch('deepcell.datasets.SegmentationDataset._get_data', mock_get_data)

        dataset = SegmentationDataset('', '')

        # Create test data to load
        shape = (10, 10, 10, 1)
        split = 'test'

        ncols = 6
        nrows = shape[0]
        tmp_meta = np.array([[''] * ncols] * (nrows + 1))

        np.savez_compressed(
            os.path.join(str(tmpdir), f'{split}.npz'),
            X=np.zeros(shape),
            y=np.zeros(shape),
            meta=tmp_meta)

        X, y, meta = dataset.load_data(split=split)
        assert X.shape == shape
        assert y.shape == shape
        assert isinstance(meta, pd.DataFrame)
        assert meta.shape == (nrows, ncols)


class TestTrackingDataset:
    def test_load_data(self, mocker, tmpdir):
        def mock_get_data(self):
            return str(tmpdir)
        mocker.patch('deepcell.datasets.TrackingDataset._get_data', mock_get_data)

        dataset = TrackingDataset('', '')

        # Create test data
        shape = (2, 10, 30, 30, 1)
        split = 'test'
        save_trks(
            os.path.join(str(tmpdir), f'{split}.trks'),
            [dict(), dict()],
            np.zeros(shape),
            np.ones(shape)
        )

        X, y, lineages = dataset.load_data(split=split)
        assert X.shape == shape
        assert y.shape == shape
        assert len(lineages) == 2
        assert isinstance(lineages[0], dict)

    def test_load_source_metadata(self, mocker, tmpdir):
        def mock_get_data(self):
            return str(tmpdir)
        mocker.patch('deepcell.datasets.TrackingDataset._get_data', mock_get_data)

        dataset = TrackingDataset('', '')

        # Save source metadata
        meta = {split: [[''] * 6] * 10 for split in ['train', 'test', 'val']}
        np.savez_compressed(
            os.path.join(str(tmpdir), 'data-source.npz'),
            **meta
        )

        df = dataset.load_source_metadata()
        for split in ['train', 'test', 'val']:
            assert split in df['split'].unique()
