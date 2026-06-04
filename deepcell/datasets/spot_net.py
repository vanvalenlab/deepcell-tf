import os

import numpy as np
import pandas as pd
from PIL import Image

from deepcell.datasets.dataset import SpotsDataset, Dataset


VERSIONS = {
    "1.0": {
        "url": "data/spotnet/SpotNet-v1_0.zip",
        "file_hash": "ad7ba11bffa242e36bd51b59f5f0abd3"
    },
    "1.1": {
        "url": "data/spotnet/SpotNet-v1_1.zip",
        "file_hash": "43691bd2d19b49c7832edb198468a4ab"
    }
}

SAMPLE_URL = "https://deepcell-data.s3.us-west-1.amazonaws.com/spot_detection/SpotNetExampleData-v1_0.zip"
SAMPLE_HASH = "bb8675da94e34805a8853b029b74e61a"

class SpotNet(SpotsDataset):
    def __init__(self, version="1.1"):
        """
        The SpotNet dataset is composed of a train, val, and test split of raw fluorescent
        spot images and coordinate spot annotations.
            - The train split is composed of 849 images, each of which are 128x128 pixels.
            - The val split is composed of 95 images, each of which are 128x128 pixels.
            - The test split is composed of 94 images, each of which are 128x128 pixels.
        See Laubscher et al. (2023) for details on image sources.

        Change Log
            - SpotNet 1.0 (Aug 2023): The original dataset used for all experiments in
              Laubscher et al. (2023)
            - SpotNet 1.1 (Jan 2024): The updated dataset, now including Airlocalize
              annotations to create consensus annotations

        Args:
            version (str, optional): Defaults to 1.1

        Example:
            >>> spotnet = SpotNet(version='1.1')  # doctest: +SKIP
            >>> X_val, y_val = spotnet.load_data(split='val')  # doctest: +SKIP

        Raises:
            ValueError: Requested version is not included in available versions
        """
        if version not in VERSIONS:
            raise ValueError(f"Requested version {version} is not included in available "
                             f"versions {list(VERSIONS.keys())}")

        self.version = version

        super().__init__(
            url=VERSIONS[version]["url"],
            file_hash=VERSIONS[version]["file_hash"],
            secure=True,
        )

class SpotNetExampleData(Dataset):
    def __init__(self):
        super().__init__(
            url=SAMPLE_URL,
            file_hash=SAMPLE_HASH,
            secure=False
        )

    def load_data(self, file='MERFISH_example'):
        """Load the specified example file required for the Polaris example notebooks.

        Args:
            file (:obj:`str`, optional):
                Data split to load from ``['seqFISH_example', 'MERFISH_example',
                'MERFISH_output', 'MERFISH_codebook']``. Defaults to ``'MERFISH_example'``.

        Raises:
            ValueError: Split must be one of ``['seqFISH example', 'MERFISH example',
                'MERFISH output', 'MERFISH codebook']``
        """
        if file not in ['seqFISH_example', 'MERFISH_example', 'MERFISH_output',
                        'MERFISH_codebook']:
            raise ValueError('Split must be one of seqFISH_example, MERFISH_example, '
                             'MERFISH_output, MERFISH_codebook')

        if file == 'seqFISH_example':
            fpath = os.path.join(self.path, f"{file}.tif")
            return self._load_tif(fpath)

        if file == 'MERFISH_example':
            fpath = os.path.join(self.path, f"{file}.npz")
            return self._load_npz(fpath)

        else:
            fpath = os.path.join(self.path, f"{file}.csv")
            return self._load_csv(fpath)
        
    def _load_tif(self, fpath):
        data = Image.open(fpath)
        data = np.array(data)
        data = np.expand_dims(data, axis=[0,-1])

        return data
    
    def _load_npz(self, fpath):
        data = np.load(fpath)
        
        spots_image = data['spots_image']
        segmentation_image = data['segmentation_image']

        return spots_image, segmentation_image
    
    def _load_csv(self, fpath):
        data = pd.read_csv(fpath, index_col=0)

        return data
