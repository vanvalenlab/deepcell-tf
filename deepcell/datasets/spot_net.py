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
from PIL import Image

from deepcell.datasets.dataset import SpotsDataset, Dataset


VERSIONS = {
    "1.0": {
        "url": "data/spotnet/SpotNet-v1_0.zip",
        "file_hash": "ad7ba11bffa242e36bd51b59f5f0abd3"
    },
}

SAMPLE_URL = "https://deepcell-data.s3.us-west-1.amazonaws.com/spot_detection/SpotNetExampleData-v1_0.zip"
SAMPLE_HASH = "bb8675da94e34805a8853b029b74e61a"

class SpotNet(SpotsDataset):
    def __init__(self, version="1.0"):
        """
        The SpotNet dataset is composed of a train, val, and test split of raw fluorescent
        spot images and coordinate spot annotations.
            - The train split is composed of 838 images, each of which are 128x128 pixels.
            - The val split is composed of 94 images, each of which are 128x128 pixels.
            - The test split is composed of 100 images, each of which are 128x128 pixels.
        See Laubscher et al. (2023) for details on image sources.

        This dataset is licensed under a modified Apache license for non-commercial academic
        use only
        http://www.github.com/vanvalenlab/deepcell-tf/LICENSE

        Change Log
            - SpotNet 1.0 (Aug 2023): The original dataset used for all experiments in
              Laubscher et al. (2023)

        Args:
            version (str, optional): Defaults to 1.0

        Example:
            >>> spotnet = SpotNet(version='1.0')  # doctest: +SKIP
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
