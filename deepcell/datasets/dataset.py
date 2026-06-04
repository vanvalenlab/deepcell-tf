"""Builtin Datasets"""


import abc
import os

import numpy as np
import pandas as pd
from tensorflow.keras.utils import get_file

from deepcell.utils import fetch_data, extract_archive

from deepcell_tracking.trk_io import load_trks


class Dataset(abc.ABC):
    def __init__(self, url, file_hash, secure=False):
        """General class for downloading datasets from S3.

        Args:
            url (str): URL of dataset in S3 or bucket path for secure data
            file_hash (str): md5hash for checking validity of cached file.
            secure (bool): True if the dataset requires a deepcell api key for download.
                Default False
        """
        self.cache_dir = os.path.expanduser(os.path.join("~", ".deepcell"))
        self.cache_subdir = "datasets"
        self.data_dir = os.path.join(self.cache_dir, self.cache_subdir)
        self.url = url
        self.secure = secure
        self.file_hash = file_hash

        self.path = self._get_data()

    def _get_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        elif not os.path.isdir(self.data_dir):
            raise OSError(f"{self.data_dir} exists but is not a directory")

        # Download data and set the path to the extracted dataset
        if self.secure:
            path = fetch_data(
                asset_key=self.url,
                cache_subdir=self.cache_subdir,
                file_hash=self.file_hash,
            )
            # For some reason this function fails inside of fetch_data but works outside
            extract_archive(path, self.data_dir)
        else:
            path = get_file(
                origin=self.url,
                file_hash=self.file_hash,
                extract=True,
                cache_dir=self.cache_dir,
                cache_subdir=self.cache_subdir,
            )

        # Strip archive extension
        if str(path).endswith('zip') or str(path).endswith('tar.gz'):
            path = os.path.splitext(path)[0]

        return path

    @abc.abstractmethod
    def load_data(self):
        """Loading function that should be implemented by the specific subclass
        """
        raise NotImplementedError


class TrackingDataset(Dataset):
    def load_data(self, split="val"):
        """Load the specified subset of the tracking dataset

        Args:
            split (:obj:`str`, optional): Data split to load from [train, test, val]. Default val.

        Returns:
            X: np.array of raw data
            y: np. array of nuclear segmentation masks
            lineages: list of lineage dictionaries

        Raises:
            ValueError: Split must be one of train, test, val
        """
        if split not in ['train', 'test', 'val']:
            raise ValueError('Split must be one of train, test, val')

        return self._load_data(os.path.join(self.path, f"{split}.trks"))

    def _load_data(self, fpath):
        data = load_trks(fpath)

        X = data["X"]
        y = data["y"]
        lineages = data["lineages"]

        return X, y, lineages

    def load_source_metadata(self):
        """Loads a pandas dataframe containing experimental metadata for each batch

        Returns:
            pd.DataFrame
        """
        data_source = np.load(
            os.path.join(self.path, "data-source.npz"), allow_pickle=True
        )

        columns = [
            "filename",
            "experiment",
            "pixel_size",
            "screening_passed",
            "time_step",
            "specimen",
        ]
        splits = list(data_source.keys())

        df = pd.concat(
            [pd.DataFrame(data_source[s], columns=columns) for s in splits], keys=splits
        )
        df = df.reset_index(0).rename(columns={'level_0': 'split'})

        return df


class SegmentationDataset(Dataset):
    def load_data(self, split="val"):
        """Load the specified subset of the segmentation dataset

        Args:
            split (:obj:`str`, optional): Data split to load from [train, test, val]. Default val.

        Returns:
            X - np.array of raw data
            y - np.array of segmentation masks
            meta - np.array of metadata if available, default None

        Raises:
            ValueError: Split must be one of train, test, val
        """
        if split not in ['train', 'test', 'val']:
            raise ValueError('Split must be one of train, test, val')

        fpath = os.path.join(self.path, f"{split}.npz")

        return self._load_data(fpath)

    def _load_data(self, fpath):
        data = np.load(fpath, allow_pickle=True)

        X = data["X"]
        y = data["y"]
        meta = data.get("meta")

        if meta is not None:
            meta = pd.DataFrame(meta[1:], columns=meta[0])

        return X, y, meta
    

class SpotsDataset(Dataset):
    def load_data(self, split="test"):
        """Load the specified subset of the segmentation dataset

        Args:
            split (:obj:`str`, optional): Data split to load from [train, val, test]. Default test.

        Returns:
            X: np.array of raw data
            y: np.array of segmentation masks

        Raises:
            ValueError: Split must be one of train, test, val
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('Split must be one of train, val, test')

        fpath = os.path.join(self.path, f"{split}.npz")

        return self._load_data(fpath)
    
    def _load_data(self, fpath):
        data = np.load(fpath, allow_pickle=True)

        X = data["X"]
        y = data["y"]

        return X, y
