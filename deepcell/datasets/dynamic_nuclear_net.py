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
"""Timelapse datasets of a nuclear label including the raw images and
ground truth segmentation masks annotated to track cell lineages"""

from deepcell.datasets import TrackingDataset, SegmentationDataset


VERSIONS_SEG = {
    "1.0": {
        "url": "data/dynamic-nuclear-net/DynamicNuclearNet-segmentation-v1_0.zip",
        "file_hash": "dcf84d150c071aedb6749084a51ddf58"  # md5
    }
}
VERSIONS_TRK = {
    "1.0": {
        "url": "data/dynamic-nuclear-net/DynamicNuclearNet-tracking-v1_0.zip",
        "file_hash": "e13ffc07fdf71f7d327e35bbdfe9bf69"  # md5
    }
}


class DynamicNuclearNetSegmentation(SegmentationDataset):
    def __init__(self, version="1.0"):
        """This dataset contains the segmentation portion of the DynamicNuclearNet dataset

        This dataset is licensed under a modified Apache license for non-commercial academic
        use only
        http://www.github.com/vanvalenlab/deepcell-tf/LICENSE

        Change Log
            - DynamicNuclearNet 1.0 (June 2023): The original dataset used for all experiments in
              Schwartz et al. 2023

        Args:
            version (str, optional): Default 1.0

        Example:
            >>>dnn_seg = DynamicNuclearNetSegmentation(version='1.0')
            >>>X_val, y_val, meta_val = dnn_seg.load_data(split='val')

        Raises:
            ValueError: Requested version is not included in available versions
        """
        if version not in VERSIONS_SEG:
            raise ValueError(f"Requested version {version} is included in available "
                             f"versions {list(VERSIONS_SEG.keys())}")

        self.version = version

        super().__init__(
            url=VERSIONS_SEG[version]["url"],
            file_hash=VERSIONS_SEG[version]["file_hash"],
            secure=True,
        )


class DynamicNuclearNetTracking(TrackingDataset):
    def __init__(self, version="1.0"):
        """This dataset contains the tracking portion of the DynamicNuclearNet dataset.
        Each batch of the dataset contains three components
        - X: raw fluorescent nuclear data
        - y: nuclear segmentation masks
        - lineages: lineage records including the cell id, frames present and division
          links from parent to daughter cells

        This dataset is licensed under a modified Apache license for non-commercial academic
        use only
        http://www.github.com/vanvalenlab/deepcell-tf/LICENSE

        Change Log
            - DynamicNuclearNet 1.0 (June 2023): The original dataset used for all experiments in
              Schwartz et al. 2023

        Args:
            version (str, optional): Default 1.0

        Example:
            >>>dnn_trk = DynamicNuclearNetTracking(version='1.0')
            >>>X_val, y_val, lineage_val = dnn_seg.load_data(split='val')
            >>>data_source = dnn_seg.load_source_metadata()

        Raises:
            ValueError: Requested version is not included in available versions
        """
        if version not in VERSIONS_TRK:
            raise ValueError(f"Requested version {version} is included in available "
                             f"versions {list(VERSIONS_SEG.keys())}")

        self.version = version

        super().__init__(
            url=VERSIONS_TRK[version]["url"],
            file_hash=VERSIONS_TRK[version]["file_hash"],
            secure=True,
        )
