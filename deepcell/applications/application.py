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
"""Base class for applications"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import timeit

import numpy as np

from deepcell_toolbox.utils import resize, tile_image, untile_image


class Application(object):
    """Application object that takes a model with weights
    and manages predictions

    Args:
        model (tensorflow.keras.Model): ``tf.keras.Model``
            with loaded weights.
        model_image_shape (tuple): Shape of input expected by ``model``.
        dataset_metadata (str or dict): Metadata for the data that
            ``model`` was trained on.
        model_metadata (str or dict): Training metadata for ``model``.
        model_mpp (float): Microns per pixel resolution of the
            training data used for ``model``.
        preprocessing_fn (function): Pre-processing function to apply
            to data prior to prediction.
        postprocessing_fn (function): Post-processing function to apply
            to data after prediction.
            Must accept an input of a list of arrays and then
            return a single array.
        format_model_output_fn (function): Convert model output
            from a list of matrices to a dictionary with keys for
            each semantic head.

    Raises:
        ValueError: ``preprocessing_fn`` must be a callable function
        ValueError: ``postprocessing_fn`` must be a callable function
        ValueError: ``model_output_fn`` must be a callable function
    """

    def __init__(self,
                 model,
                 model_image_shape=(128, 128, 1),
                 model_mpp=0.65,
                 preprocessing_fn=None,
                 postprocessing_fn=None,
                 format_model_output_fn=None,
                 dataset_metadata=None,
                 model_metadata=None):

        self.model = model

        self.model_image_shape = model_image_shape
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 1

        self.required_channels = self.model_image_shape[-1]

        self.model_mpp = model_mpp
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        self.format_model_output_fn = format_model_output_fn
        self.dataset_metadata = dataset_metadata
        self.model_metadata = model_metadata

        self.logger = logging.getLogger(self.__class__.__name__)

        # Test that pre and post processing functions are callable
        if self.preprocessing_fn is not None and not callable(self.preprocessing_fn):
            raise ValueError('Preprocessing_fn must be a callable function.')
        if self.postprocessing_fn is not None and not callable(self.postprocessing_fn):
            raise ValueError('Postprocessing_fn must be a callable function.')
        if self.format_model_output_fn is not None and not callable(self.format_model_output_fn):
            raise ValueError('Format_model_output_fn must be a callable function.')

    def predict(self, x):
        raise NotImplementedError

    def _resize_input(self, image, image_mpp):
        """Checks if there is a difference between image and model resolution
        and resizes if they are different. Otherwise returns the unmodified
        image.

        Args:
            image (numpy.array): Input image to resize.
            image_mpp (float): Microns per pixel for the ``image``.

        Returns:
            numpy.array: Input image resized if necessary to match ``model_mpp``
        """
        # Don't scale the image if mpp is the same or not defined
        if image_mpp not in {None, self.model_mpp}:
            shape = image.shape
            scale_factor = image_mpp / self.model_mpp
            new_shape = (int(shape[1] * scale_factor),
                         int(shape[2] * scale_factor))
            image = resize(image, new_shape, data_format='channels_last')
            self.logger.debug('Resized input from %s to %s', shape, new_shape)

        return image

    def _preprocess(self, image, **kwargs):
        """Preprocess ``image`` if ``preprocessing_fn`` is defined.
        Otherwise return ``image`` unmodified.

        Args:
            image (numpy.array): 4D stack of images
            kwargs (dict): Keyword arguments for ``preprocessing_fn``.

        Returns:
            numpy.array: The pre-processed ``image``.
        """
        if self.preprocessing_fn is not None:
            t = timeit.default_timer()
            self.logger.debug('Pre-processing data with %s and kwargs: %s',
                              self.preprocessing_fn.__name__, kwargs)

            image = self.preprocessing_fn(image, **kwargs)

            self.logger.debug('Pre-processed data with %s in %s s',
                              self.preprocessing_fn.__name__,
                              timeit.default_timer() - t)

        return image

    def _tile_input(self, image, pad_mode='constant'):
        """Tile the input image to match shape expected by model
        using the ``deepcell_toolbox`` function.

        Only supports 4D images.

        Args:
            image (numpy.array): Input image to tile
            pad_mode (str): The padding mode, one of "constant" or "reflect".

        Raises:
            ValueError: Input images must have only 4 dimensions

        Returns:
            (numpy.array, dict): Tuple of tiled image and dict of tiling
            information.
        """
        if len(image.shape) != 4:
            raise ValueError('deepcell_toolbox.tile_image only supports 4d images.'
                             'Image submitted for predict has {} dimensions'.format(
                                 len(image.shape)))

        # Check difference between input and model image size
        x_diff = image.shape[1] - self.model_image_shape[0]
        y_diff = image.shape[2] - self.model_image_shape[1]

        # Check if the input is smaller than model image size
        if x_diff < 0 or y_diff < 0:
            # Calculate padding
            x_diff, y_diff = abs(x_diff), abs(y_diff)
            x_pad = (x_diff // 2, x_diff // 2 + 1) if x_diff % 2 else (x_diff // 2, x_diff // 2)
            y_pad = (y_diff // 2, y_diff // 2 + 1) if y_diff % 2 else (y_diff // 2, y_diff // 2)

            tiles = np.pad(image, [(0, 0), x_pad, y_pad, (0, 0)], 'reflect')
            tiles_info = {'padding': True,
                          'x_pad': x_pad,
                          'y_pad': y_pad}
        # Otherwise tile images larger than model size
        else:
            # Tile images, needs 4d
            tiles, tiles_info = tile_image(image, model_input_shape=self.model_image_shape,
                                           stride_ratio=0.75, pad_mode=pad_mode)

        return tiles, tiles_info

    def _postprocess(self, image, **kwargs):
        """Applies postprocessing function to image if one has been defined.
        Otherwise returns unmodified image.

        Args:
            image (numpy.array or list): Input to postprocessing function
                either an ``numpy.array`` or list of ``numpy.arrays``.

        Returns:
            numpy.array: labeled image
        """
        if self.postprocessing_fn is not None:
            t = timeit.default_timer()
            self.logger.debug('Post-processing results with %s and kwargs: %s',
                              self.postprocessing_fn.__name__, kwargs)

            image = self.postprocessing_fn(image, **kwargs)

            # Restore channel dimension if not already there
            if len(image.shape) == self.required_rank - 1:
                image = np.expand_dims(image, axis=-1)

            self.logger.debug('Post-processed results with %s in %s s',
                              self.postprocessing_fn.__name__,
                              timeit.default_timer() - t)

        elif isinstance(image, list) and len(image) == 1:
            image = image[0]

        return image

    def _untile_output(self, output_tiles, tiles_info):
        """Untiles either a single array or a list of arrays
        according to a dictionary of tiling specs

        Args:
            output_tiles (numpy.array or list): Array or list of arrays.
            tiles_info (dict): Tiling specs output by the tiling function.

        Returns:
            numpy.array or list: Array or list according to input with untiled images
        """
        # If padding was used, remove padding
        if tiles_info.get('padding', False):
            def _process(im, tiles_info):
                x_pad, y_pad = tiles_info['x_pad'], tiles_info['y_pad']
                out = im[:, x_pad[0]:-x_pad[1], y_pad[0]:-y_pad[1], :]
                return out
        # Otherwise untile
        else:
            def _process(im, tiles_info):
                out = untile_image(im, tiles_info, model_input_shape=self.model_image_shape)
                return out

        if isinstance(output_tiles, list):
            output_images = [_process(o, tiles_info) for o in output_tiles]
        else:
            output_images = _process(output_tiles, tiles_info)

        return output_images

    def _format_model_output(self, output_images):
        """Applies formatting function the output from the model if one was
        provided. Otherwise, returns the unmodified model output.

        Args:
            output_images: stack of untiled images to be reformatted

        Returns:
            dict or list: reformatted images stored as a dict, or input
            images stored as list if no formatting function is specified.
        """
        if self.format_model_output_fn is not None:
            formatted_images = self.format_model_output_fn(output_images)
            return formatted_images
        else:
            return output_images

    def _resize_output(self, image, original_shape):
        """Rescales input if the shape does not match the original shape
        excluding the batch and channel dimensions.

        Args:
            image (numpy.array): Image to be rescaled to original shape
            original_shape (tuple): Shape of the original input image

        Returns:
            numpy.array: Rescaled image
        """
        if not isinstance(image, list):
            image = [image]

        for i in range(len(image)):
            img = image[i]
            # Compare x,y based on rank of image
            if len(img.shape) == 4:
                same = img.shape[1:-1] == original_shape[1:-1]
            elif len(img.shape) == 3:
                same = img.shape[1:] == original_shape[1:-1]
            else:
                same = img.shape == original_shape[1:-1]

            # Resize if same is false
            if not same:
                # Resize function only takes the x,y dimensions for shape
                new_shape = original_shape[1:-1]
                img = resize(img, new_shape,
                             data_format='channels_last',
                             labeled_image=True)
            image[i] = img

        if len(image) == 1:
            image = image[0]

        return image

    def _run_model(self,
                   image,
                   batch_size=4,
                   pad_mode='constant',
                   preprocess_kwargs={}):
        """Run the model to generate output probabilities on the data.

        Args:
            image (numpy.array): Image with shape ``[batch, x, y, channel]``
            batch_size (int): Number of images to predict on per batch.
            pad_mode (str): The padding mode, one of "constant" or "reflect".
            preprocess_kwargs (dict): Keyword arguments to pass to
                the preprocessing function.

        Returns:
            numpy.array: Model outputs
        """
        # Preprocess image if function is defined
        image = self._preprocess(image, **preprocess_kwargs)

        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = self._tile_input(image, pad_mode=pad_mode)

        # Run images through model
        t = timeit.default_timer()
        output_tiles = self.model.predict(tiles, batch_size=batch_size)
        self.logger.debug('Model inference finished in %s s',
                          timeit.default_timer() - t)

        # Untile images
        output_images = self._untile_output(output_tiles, tiles_info)

        # restructure outputs into a dict if function provided
        formatted_images = self._format_model_output(output_images)

        return formatted_images

    def _predict_segmentation(self,
                              image,
                              batch_size=4,
                              image_mpp=None,
                              pad_mode='constant',
                              preprocess_kwargs={},
                              postprocess_kwargs={}):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``. Additional empty dimensions can be added
        using ``np.expand_dims``.

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
            ValueError: Input data must match required rank, calculated as one
                dimension more (batch dimension) than expected by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Labeled image
        """
        # Check input size of image
        if len(image.shape) != self.required_rank:
            raise ValueError('Input data must have {} dimensions. '
                             'Input data only has {} dimensions'.format(
                                 self.required_rank, len(image.shape)))

        if image.shape[-1] != self.required_channels:
            raise ValueError('Input data must have {} channels. '
                             'Input data only has {} channels'.format(
                                 self.required_channels, image.shape[-1]))

        # Resize image, returns unmodified if appropriate
        resized_image = self._resize_input(image, image_mpp)

        # Generate model outputs
        output_images = self._run_model(
            image=resized_image, batch_size=batch_size,
            pad_mode=pad_mode, preprocess_kwargs=preprocess_kwargs
        )

        # Postprocess predictions to create label image
        label_image = self._postprocess(output_images, **postprocess_kwargs)

        # Resize label_image back to original resolution if necessary
        label_image = self._resize_output(label_image, image.shape)

        return label_image
