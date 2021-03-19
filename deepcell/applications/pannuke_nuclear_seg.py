"""Nuclear segmentation application for Pannuke Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from deepcell_toolbox.processing import normalize
from deepcell_toolbox.deep_watershed import deep_watershed

from deepcell.applications import Application


MODEL_PATH = ('./pannuc_panopticnet.h5')


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
        'name': 'Pannuke Dataset',
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 8,
        'lr': 1e-4,
        'lr_decay': 0.99,
        'training_seed': 808,
        'n_epochs': 5,
        'training_steps_per_epoch': 1987,
        'validation_steps_per_epoch': 253
    }

    def __init__(self, model=None):

        if model is None:
            archive_path = tf.keras.utils.get_file(
                'NuclearSegmentation.tgz', MODEL_PATH,
                file_hash='7fff56a59f453252f24967cfe1813abd',
                extract=True, cache_subdir='models'
            )
            model_path = os.path.splitext(archive_path)[0]
            model = tf.keras.models.load_model(model_path)

        super(NuclearSegmentation, self).__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=0.65,
            preprocessing_fn=normalize,
            postprocessing_fn=deep_watershed,
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
            postprocess_kwargs = {
                'min_distance': 10,
                'detection_threshold': 0.1,
                'distance_threshold': 0.01,
                'exclude_border': False,
                'small_objects_threshold': 0
            }

        return self._predict_segmentation(
            image,
            batch_size=batch_size,
            image_mpp=image_mpp,
            pad_mode=pad_mode,
            preprocess_kwargs=preprocess_kwargs,
            postprocess_kwargs=postprocess_kwargs)