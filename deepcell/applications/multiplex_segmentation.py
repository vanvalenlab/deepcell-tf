# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""Multiplex segmentation application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras.utils.data_utils import get_file

from deepcell_toolbox.deep_watershed import deep_watershed_subcellular, format_output_multiplex
from deepcell_toolbox.processing import phase_preprocess

from deepcell.applications import Application
from deepcell.model_zoo import PanopticNet


WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/Multiplex_Segmentation_20200714_subcellular.h5')


class MultiplexSegmentation(Application):
    """Loads a `deepcell.model_zoo.PanopticNet` model for multiplex segmentation
    with pretrained weights.
    The `predict` method handles prep and post processing steps to return a labeled image.

    Example:

    .. nbinput:: ipython3

        from skimage.io import imread
        from deepcell.applications import MultiplexSegmentation

        im1 = imread('TNBC_DNA.tiff')
        im2 = imread('TNBC_Membrane.tiff')
        im1.shape

    .. nboutput::

        (1024, 1024)

    .. nbinput:: ipython3

        # Combined together and expand to 4D
        im = np.stack((im1, im2), axis=-1)
        im = np.expand_dims(im,0)
        im.shape

    .. nboutput::

        (1, 1024, 1024, 2)

    .. nbinput:: ipython3

        app = MultiplexSegmentation(use_pretrained_weights=True)
        labeled_image = app.predict(image)

    .. nboutput::

    Args:
        use_pretrained_weights (bool, optional): Loads pretrained weights. Defaults to True.
        model_image_shape (tuple, optional): Shape of input data expected by model.
            Defaults to `(256, 256, 2)`
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': '20200315_IF_Training_6.npz',
        'other': 'Pooled whole-cell data across tissue types'
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 1,
        'lr': 1e-5,
        'lr_decay': 0.99,
        'training_seed': 0,
        'n_epochs': 30,
        'training_steps_per_epoch': 1739 // 1,
        'validation_steps_per_epoch': 193 // 1
    }

    def __init__(self,
                 use_pretrained_weights=True,
                 model_image_shape=(256, 256, 2)):

        whole_cell_classes = [1, 1, 2, 3]
        nuclear_classes = [1, 1, 2, 3]
        num_semantic_classes = whole_cell_classes + nuclear_classes
        num_semantic_heads = len(num_semantic_classes)

        model = PanopticNet('resnet50',
                            input_shape=model_image_shape,
                            norm_method=None,
                            num_semantic_heads=num_semantic_heads,
                            num_semantic_classes=num_semantic_classes,
                            location=True,
                            include_top=True,
                            use_imagenet=False)

        if use_pretrained_weights:
            weights_path = get_file(
                os.path.basename(WEIGHTS_PATH),
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a2f26a7b5cc9a86d68e65a7bd27c32d0'
            )

            model.load_weights(weights_path)
        else:
            weights_path = None

        super(MultiplexSegmentation, self).__init__(model,
                                                    model_image_shape=model_image_shape,
                                                    model_mpp=0.5,
                                                    preprocessing_fn=phase_preprocess,
                                                    postprocessing_fn=deep_watershed_subcellular,
                                                    format_model_output_fn=format_output_multiplex,
                                                    dataset_metadata=self.dataset_metadata,
                                                    model_metadata=self.model_metadata)

    def predict(self,
                image,
                batch_size=4,
                image_mpp=None,
                preprocess_kwargs={},
                compartment='whole-cell',
                postprocess_kwargs_whole_cell={},
                postprocess_kwargs_nuclear={}):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions `[batch, x, y, channel]`. Additional
        empty dimensions can be added using `np.expand_dims`

        Args:
            image (np.array): Input image with shape `[batch, x, y, channel]`
            batch_size (int, optional): Number of images to predict on per batch. Defaults to 4.
            image_mpp (float, optional): Microns per pixel for the input image. Defaults to None.
            preprocess_kwargs (dict, optional): Kwargs to pass to preprocessing function.
                Defaults to {}.
            compartment (string): Specify type of segmentation to predict. Must be one of
                [whole-cell, nuclear, both]
            postprocess_kwargs_whole_cell (dict, optional): Kwargs to pass to postprocessing
                function for whole_cell prediction. Defaults to {}.
            postprocess_kwargs_nuclear (dict, optional): Kwargs to pass to postprocessing
                function for nuclear prediction. Defaults to {}.

        Raises:
            ValueError: Input data must match required rank of the application, calculated as
                one dimension more (batch dimension) than expected by the model

            ValueError: Input data must match required number of channels of application

        Returns:
            np.array: Labeled image
            np.array: Model output
        """

        # create dict to hold all of the post-processing kwargs
        postprocess_kwargs = {'whole_cell_kwargs': postprocess_kwargs_whole_cell,
                              'nuclear_kwargs': postprocess_kwargs_nuclear,
                              'compartment': compartment}

        return self._predict_segmentation(image,
                                          batch_size=batch_size,
                                          image_mpp=image_mpp,
                                          preprocess_kwargs=preprocess_kwargs,
                                          postprocess_kwargs=postprocess_kwargs)
