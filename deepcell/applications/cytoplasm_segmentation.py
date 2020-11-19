# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
"""Cytoplasmic segmentation application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.keras.utils import get_file

from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell_toolbox.processing import phase_preprocess

from deepcell.applications import Application
from deepcell.model_zoo import PanopticNet


WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/general_cyto_named.h5')


class CytoplasmSegmentation(Application):
    """Loads a :mod:`deepcell.model_zoo.panopticnet.PanopticNet` model
    for cytoplasm segmentation with pretrained weights.

    The ``predict`` method handles prep and post processing steps
    to return a labeled image.

    Example:

    .. code-block:: python

        from skimage.io import imread
        from deepcell.applications import CytoplasmSegmentation

        # Load the image
        im = imread('HeLa_cytoplasm.png')

        # Expand image dimensions to rank 4
        im = np.expand_dims(im, axis=-1)
        im = np.expand_dims(im, axis=0)

        # Create the application
        app = CytoplasmSegmentation(use_pretrained_weights=True)

        # create the lab
        labeled_image = app.predict(image)

    Args:
        use_pretrained_weights (bool): Whether to load pretrained weights.
        model_image_shape (tuple): Shape of input expected by ``model``.
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': 'general_cyto',
        'other': 'Pooled phase and fluorescent cytoplasm data - computationally curated'
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 2,
        'lr': 1e-4,
        'lr_decay': 0.95,
        'training_seed': 0,
        'n_epochs': 8,
        'training_steps_per_epoch': 7899 // 2,
        'validation_steps_per_epoch': 1973 // 2
    }

    def __init__(self,
                 use_pretrained_weights=True,
                 model_image_shape=(128, 128, 1)):

        model = PanopticNet('resnet50',
                            input_shape=model_image_shape,
                            norm_method='whole_image',
                            num_semantic_heads=2,
                            num_semantic_classes=[1, 1],
                            location=True,
                            include_top=True,
                            lite=True,
                            use_imagenet=use_pretrained_weights,
                            interpolation='bilinear')

        if use_pretrained_weights:
            weights_path = get_file(
                os.path.basename(WEIGHTS_PATH),
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='104a7d7884c80c37d2bce6d1c3a17c7a'
            )

            model.load_weights(weights_path, by_name=True)
        else:
            weights_path = None

        super(CytoplasmSegmentation, self).__init__(
            model,
            model_image_shape=model_image_shape,
            model_mpp=0.65,
            preprocessing_fn=phase_preprocess,
            postprocessing_fn=deep_watershed,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)

    def predict(self,
                image,
                batch_size=4,
                image_mpp=None,
                preprocess_kwargs={},
                postprocess_kwargs={}):
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
        return self._predict_segmentation(
            image,
            batch_size=batch_size,
            image_mpp=image_mpp,
            preprocess_kwargs=preprocess_kwargs,
            postprocess_kwargs=postprocess_kwargs)
