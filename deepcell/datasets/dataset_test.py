import os

import numpy as np
import pandas as pd

from deepcell_tracking.trk_io import save_trks

from deepcell.datasets.dataset import SegmentationDataset, TrackingDataset, SpotsDataset


class TestSegmentationDataset:
    def test_no_meta(self, tmpdir, mocker):
        def mock_get_data(self):
            return str(tmpdir)
        mocker.patch('deepcell.datasets.dataset.SegmentationDataset._get_data', mock_get_data)

        dataset = SegmentationDataset('', '')

        # Create test data to load
        shape = (1, 10, 10, 1)
        split = 'test'
        np.savez_compressed(
            os.path.join(str(tmpdir), f'{split}.npz'),
            X=np.zeros(shape),
            y=np.zeros(shape))

        X, y, meta = dataset.load_data(split=split)
        assert X.shape == shape
        assert y.shape == shape
        assert meta is None

    def test_meta(self, tmpdir, mocker):
        def mock_get_data(self):
            return str(tmpdir)
        mocker.patch('deepcell.datasets.dataset.SegmentationDataset._get_data', mock_get_data)

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
        mocker.patch('deepcell.datasets.dataset.TrackingDataset._get_data', mock_get_data)

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
        mocker.patch('deepcell.datasets.dataset.TrackingDataset._get_data', mock_get_data)

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


class TestSpotsDataset:
    def test_load_data(self, mocker, tmpdir):
        def mock_get_data(self):
            return str(tmpdir)
        mocker.patch('deepcell.datasets.dataset.SpotsDataset._get_data', mock_get_data)

        dataset = SpotsDataset('', '')

        # Create test data
        X_shape = (1, 128, 128, 1)
        y_shape = (1, 10, 2)
        split = 'test'
        np.savez_compressed(
            os.path.join(str(tmpdir), f'{split}.npz'),
            X=np.zeros(X_shape),
            y=np.zeros(y_shape))

        X, y = dataset.load_data(split=split)
        assert X.shape == X_shape
        assert y.shape == y_shape
