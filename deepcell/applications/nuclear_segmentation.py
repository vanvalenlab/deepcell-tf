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
"""Nuclear segmentation application"""


from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from deepcell_toolbox.processing import histogram_normalization
from deepcell_toolbox.deep_watershed import deep_watershed

from deepcell.applications import Application
from deepcell.utils import fetch_data, extract_archive


@dataclass
class Config:
    model_key: str
    model_hash: str
    radius: int
    maxima_threshold: float
    interior_threshold: float
    exclude_border: bool
    small_objects_threshold: float
    min_distance: float
    model_mpp: float


CONFIGS = {
    '1.1': Config(
        model_key='models/NuclearSegmentation-8.tar.gz',
        model_hash='507be21f0e34e59adae689f58cc03ccb',
        radius=10,
        maxima_threshold=0.1,
        interior_threshold=0.01,
        exclude_border=False,
        small_objects_threshold=0,
        min_distance=10,
        model_mpp=0.65
    ),
    '1.0': Config(
        model_key='models/NuclearSegmentation-75.tar.gz',
        model_hash='efc4881db5bac23219b62486a4d877b3',
        radius=10,
        maxima_threshold=0.1,
        interior_threshold=0.01,
        exclude_border=False,
        small_objects_threshold=0,
        min_distance=10,
        model_mpp=0.65
    )
}

class NuclearSegmentation(Application):
    """Loads a :mod:`deepcell.model_zoo.panopticnet.PanopticNet` model
    for nuclear segmentation with pretrained weights.

    The ``predict`` method handles prep and post processing steps
    to return a labeled image.

    Example:

    .. code-block:: python

        from skimage.io import imread
        from deepcell.applications import NuclearSegmentation

        # Load the image
        im = imread('HeLa_nuclear.png')

        # Expand image dimensions to rank 4
        im = np.expand_dims(im, axis=-1)
        im = np.expand_dims(im, axis=0)

        # Create the application
        app = NuclearSegmentation()

        # create the lab
        labeled_image = app.predict(image)

    Args:
        model (tf.keras.Model): The model to load. If ``None``,
            a pre-trained model will be downloaded.
    """
    def __init__(self,
                 model,
                 model_mpp=0.65,
                 radius=10,
                 maxima_threshold=0.1,
                 interior_threshold=0.01,
                 exclude_border=False,
                 small_objects_threshold=0,
                 min_distance=10
                 ):

        self.postprocess_kwargs = {
            'radius': radius,
            'maxima_threshold': maxima_threshold,
            'interior_threshold': interior_threshold,
            'exclude_border': exclude_border,
            'small_objects_threshold': small_objects_threshold,
            'min_distance': min_distance
        }

        super().__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=model_mpp,
            preprocessing_fn=histogram_normalization,
            postprocessing_fn=deep_watershed)

    @classmethod
    def from_version(cls, version='1.1'):
        """Load a specified version of the model

        1.1: Updates to the nuclear segmentation model released in July 2024
        1.0: Original nuclear segmentation model released with the September 2023 Caliban preprint

        Args:
            version (:obj:`str`, optional): Defaults to '1.1'.
        """
        if version not in CONFIGS:
            raise ValueError(f'Selected version {version} is not available. '
                             f'Choose from {CONFIGS.keys()}')

        config = CONFIGS[version]

        cache_subdir = 'models'
        model_dir = Path.home() / ".deepcell" / "models"
        archive_path = fetch_data(
            asset_key=config.model_key,
            cache_subdir=cache_subdir,
            file_hash=config.model_hash
        )
        extract_archive(archive_path, model_dir)
        model_path = model_dir / 'NuclearSegmentation'
        model = tf.keras.models.load_model(model_path)

        return cls(
            model,
            model_mpp=config.model_mpp,
            radius=config.radius,
            maxima_threshold=config.maxima_threshold,
            interior_threshold=config.interior_threshold,
            exclude_border=False,
            small_objects_threshold=config.small_objects_threshold,
            min_distance=config.min_distance
        )


    def predict(self,
                image,
                batch_size=4,
                image_mpp=None,
                pad_mode='reflect',
                preprocess_kwargs=None,
                postprocess_kwargs=None):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``.

        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.
            pad_mode (str): The padding mode, one of "constant" or "reflect".
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Labeled image
        """
        if preprocess_kwargs is None:
            preprocess_kwargs = {}

        if postprocess_kwargs is None:
            postprocess_kwargs = self.postprocess_kwargs

        return self._predict_segmentation(
            image,
            batch_size=batch_size,
            image_mpp=image_mpp,
            pad_mode=pad_mode,
            preprocess_kwargs=preprocess_kwargs,
            postprocess_kwargs=postprocess_kwargs)
