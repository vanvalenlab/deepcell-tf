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


from pathlib import Path

import tensorflow as tf

from deepcell_toolbox.processing import histogram_normalization
from deepcell_toolbox.deep_watershed import deep_watershed

from deepcell.applications import Application
from deepcell.utils import fetch_data, extract_archive


MODEL_KEY = 'models/NuclearSegmentation-75.tar.gz'
MODEL_NAME = 'NuclearSegmentation'
MODEL_HASH = 'efc4881db5bac23219b62486a4d877b3'

MODEL_METADATA = {
    'crop_size': 256,
    'min_objects': 1,
    'zoom_min': 0.75,
    'epochs': 16,
    'batch_size': 16,
    'backbone': 'efficientnetv2bl',
    'lr': .0001,
    'location': True,
    'pyramid_levels': 'P1-P2-P3-P4-P5-P6-P7'
}

POSTPROCESS_KWARGS = {
    'radius': 10,
    'maxima_threshold': 0.1,
    'interior_threshold': 0.01,
    'exclude_border': False,
    'small_objects_threshold': 0,
    'min_distance': 10
}
MODEL_MPP = 0.65

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

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': 'general_nuclear_train_large',
        'other': 'Pooled nuclear data from HEK293, HeLa-S3, NIH-3T3, and RAW264.7 cells.'
    }

    #: Metadata for the model and training process
    model_metadata = MODEL_METADATA

    def __init__(self, model=None,
                 preprocessing_fn=histogram_normalization,
                 postprocessing_fn=deep_watershed):

        if model is None:
            cache_subdir = 'models'
            model_dir = Path.home() / ".deepcell" / "models"
            archive_path = fetch_data(
                asset_key=MODEL_KEY,
                cache_subdir=cache_subdir,
                file_hash=MODEL_HASH
            )
            extract_archive(archive_path, model_dir)
            model_path = model_dir / MODEL_NAME
            model = tf.keras.models.load_model(model_path)

        super().__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=MODEL_MPP,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)

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
            postprocess_kwargs = POSTPROCESS_KWARGS

        return self._predict_segmentation(
            image,
            batch_size=batch_size,
            image_mpp=image_mpp,
            pad_mode=pad_mode,
            preprocess_kwargs=preprocess_kwargs,
            postprocess_kwargs=postprocess_kwargs)
