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

from deepcell.datasets.dataset import SegmentationDataset


VERSIONS = {
    "1.1": {
        "url": "data/tissuenet/tissuenet_v1-1.zip",
        "file_hash": "cab3b8f242aaee02035557b93546d9dc"
    },
    "1.0": {
        "url": "data/tissuenet/tissuenet_1-0.zip",
        "file_hash": "f080c7732dd6de71e8e72e95a314e904"
    },
}
SAMPLE_URL = "https://deepcell-data.s3.us-west-1.amazonaws.com/multiplex/tissuenet-sample.npz"
SAMPLE_HASH = "de5b1e73373f7783fc6b11f4cb295638"


class TissueNet(SegmentationDataset):
    def __init__(self, version="1.1"):
        """
        The TissueNet dataset is composed of a train, val, and test split.
            - The train split is composed of aproximately 2600 images, each of which are 512x512
              pixels. During training, we select random crops of size 256x256 from each image as
              a form of data augmentation.
            - The val split is composed of aproximately 300 images, each of which is originally
              of size 512x512. However, because we do not perform any augmentation on the
              validation dataset during training, we reshape these 512x512 images into 256x256
              images so that no cropping is needed in order to pass them through the model.
              Finally, we make two copies of the val set at different image resolutions and
              concatenate them all together, resulting in a total of aproximately 3000 images
              of size 256x256,
            - The test split is composed of aproximately 300 images, each of which is originally
              of size 512x512. However, because the model was trained on images that are size
              256x256, we reshape these 512x512 images into 256x256 images, resulting in
              aproximately 1200 images.

        This dataset is licensed under a modified Apache license for non-commercial academic
        use only
        http://www.github.com/vanvalenlab/deepcell-tf/LICENSE

        Change Log
            - TissueNet 1.0 (July 2021): The original dataset used for all experiments in
              Greenwald, Miller at al.
            - TissueNet 1.1 (April 2022): This version of TissueNet has gone through an additional
              round of manual QC to ensure consistency in labeling across the entire dataset.

        Args:
            version (:obj:`str`, optional): Default 1.1

        Example:
            >>>tissuenet = TissueNet(version='1.1')
            >>>X_val, y_val, meta_val = tissuenet.load_data(split='val')

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


class TissueNetSample(SegmentationDataset):
    def __init__(self):
        """This is a single sample image from TissueNet"""

        super().__init__(
            url=SAMPLE_URL,
            file_hash=SAMPLE_HASH,
            secure=False
        )

    def load_data(self):
        return self._load_data(self.path)
