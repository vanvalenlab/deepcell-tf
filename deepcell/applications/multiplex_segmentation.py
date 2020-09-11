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

import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

from deepcell_toolbox.deep_watershed import deep_watershed_mibi
from deepcell_toolbox.processing import histogram_normalization, percentile_threshold

from deepcell.applications import Application
from deepcell.model_zoo import PanopticNet


WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/Multiplex_Segmentation_20200816_compartment.h5')

# processing functions


def multiplex_preprocess(image, **kwargs):
    """Preprocess input data for multiplex model

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """
    output = np.copy(image)
    threshold = kwargs.get('threshold', True)
    if threshold:
        percentile = kwargs.get('percentile', 99.9)
        output = percentile_threshold(image=output, percentile=percentile)

    normalize = kwargs.get('normalize', True)
    if normalize:
        kernel_size = kwargs.get('kernel_size', 128)
        output = histogram_normalization(image=output, kernel_size=kernel_size)

    return output


def format_output_multiplex(output_list):
    """Takes list of model outputs and formats into a dictionary for better readability

    Args:
        output_list: list of predictions from semantic heads

    Returns:
        formatted_dict: dictionary with predictions

    Raises: ValueError if model output list is not len(8)
    """

    if len(output_list) != 8:
        raise ValueError('output_list was length {}, expecting length 8'.format(len(output_list)))

    formatted_dict = {
        'whole-cell': {
            'inner-distance': output_list[0],
            'outer-distance': output_list[1],
            'fgbg-fg': output_list[2][..., :1],
            'pixelwise-interior': output_list[3][..., 1:2]
        },
        'nuclear': {
            'inner-distance': output_list[4],
            'outer-distance': output_list[5],
            'fgbg-fg': output_list[6][..., :1],
            'pixelwise-interior': output_list[7][..., 1:2]
        }
    }

    return formatted_dict


def multiplex_postprocess(model_output, compartment='whole-cell', whole_cell_kwargs=None,
                          nuclear_kwargs=None):
    """Postprocess model output to generate predictions for distinct cellular compartments

    Args:
        model_output (dict): Output from deep watershed model. A dict with a key corresponding to
            each cellular compartment with a model prediction. Each key maps to a subsequent dict
            with the following keys entries
            - inner-distance: Prediction for the inner distance transform.
            - outer-distance: Prediction for the outer distance transform
            - fgbg-fg: prediction for the foreground/background transform
            - pixelwise-interior: Prediction for the interior/border/background transform.
        compartment: which cellular compartments to generate predictions for.
            must be one of 'whole_cell', 'nuclear', 'both'
        whole_cell_kwargs (dict): Optional list of post-processing kwargs for whole-cell prediction
        nuclear_kwargs (dict): Optional list of post-processing kwargs for nuclear prediction

    Returns:
        numpy.array: Uniquely labeled mask for each compartment

    Raises:
        ValueError: for invalid compartment flag
    """

    valid_compartments = ['whole-cell', 'nuclear', 'both']

    if whole_cell_kwargs is None:
        whole_cell_kwargs = {}

    if nuclear_kwargs is None:
        nuclear_kwargs = {}

    if compartment not in valid_compartments:
        raise ValueError('Invalid compartment supplied: {}. '
                         'Must be one of {}'.format(compartment, valid_compartments))

    if compartment == 'whole-cell':
        label_images = deep_watershed_mibi(model_output=model_output['whole-cell'],
                                           **whole_cell_kwargs)
    elif compartment == 'nuclear':
        label_images = deep_watershed_mibi(model_output=model_output['nuclear'],
                                           **nuclear_kwargs)
    elif compartment == 'both':
        label_images_cell = deep_watershed_mibi(model_output=model_output['whole-cell'],
                                                **whole_cell_kwargs)

        label_images_nucleus = deep_watershed_mibi(model_output=model_output['nuclear'],
                                                   **nuclear_kwargs)

        label_images = np.concatenate((label_images_cell, label_images_nucleus), axis=-1)

    else:
        raise ValueError('Invalid compartment supplied: {}. '
                         'Must be one of {}'.format(compartment, valid_compartments))

    return label_images


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
                file_hash='ff24e821c6056cf847e58e8e52916814'
            )

            model.load_weights(weights_path)
        else:
            weights_path = None

        super(MultiplexSegmentation, self).__init__(model,
                                                    model_image_shape=model_image_shape,
                                                    model_mpp=0.5,
                                                    preprocessing_fn=multiplex_preprocess,
                                                    postprocessing_fn=multiplex_postprocess,
                                                    format_model_output_fn=format_output_multiplex,
                                                    dataset_metadata=self.dataset_metadata,
                                                    model_metadata=self.model_metadata)
    cell_defaults = {'interior_smooth': 1}
    nuc_defaults = {'interior_smooth': 1}

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
