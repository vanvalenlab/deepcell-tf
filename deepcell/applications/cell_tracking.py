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
"""A model that can detect whether 2 cells are same, different, or related."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import deepcell_tracking
from deepcell_toolbox.processing import normalize

from deepcell.applications import Application


MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
              'saved-models/TrackingModel-2.tar.gz')


class CellTracking(Application):
    """Loads a :mod:`deepcell.model_zoo.featurenet.siamese_model` model for
    object tracking with pretrained weights using a simple
    ``predict`` interface.

    Args:
        use_pretrained_weights (bool): Whether to load pretrained weights.
        model_image_shape (tuple): Shape of input data expected by model.
        neighborhood_scale_size (int): Size of the area surrounding each cell.
        birth (float): Cost of new cell in linear assignment matrix.
        death (float): Cost of cell death in linear assignment matrix.
        division (float): Cost of cell division in linear assignment matrix.
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': 'tracked_nuclear_train_large',
        'other': 'Pooled tracked nuclear data from HEK293, HeLa-S3, NIH-3T3, and RAW264.7 cells.'
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 128,
        'lr': 1e-2,
        'lr_decay': 0.99,
        'training_seed': 757,
        'n_epochs': 10,
        'training_steps_per_epoch': 5536,
        'validation_steps_per_epoch': 1427,
        'features': {'appearance', 'distance', 'neighborhood', 'regionprop'},
        'min_track_length': 9,
        'neighborhood_scale_size': 30,
        'crop_dim': 32,
    }

    def __init__(self,
                 model=None,
                 model_image_shape=(32, 32, 1),
                 birth=0.99,
                 death=0.99,
                 division=0.9,
                 track_length=9):
        self.features = {'appearance', 'distance', 'neighborhood', 'regionprop'}
        self.birth = birth
        self.death = death
        self.division = division
        self.track_length = track_length

        if model is None:
            archive_path = tf.keras.utils.get_file(
                'Tracking.tgz', MODEL_PATH,
                file_hash='06e2043b4b898c9f81baeda9b6950ce0',
                extract=True, cache_subdir='models')
            model_path = os.path.splitext(archive_path)[0]
            model = tf.keras.models.load_model(model_path)

        super(CellTracking, self).__init__(
            model,
            model_image_shape=model_image_shape,
            model_mpp=0.65,
            preprocessing_fn=None,
            postprocessing_fn=None,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)

    def predict(self, image, labels, **kwargs):
        """Using both raw image data and segmentation masks,
        track objects across all frames.

        Args:
            image (numpy.array): Raw image data.
            labels (numpy.array): Labels for ``image``, integer masks.

        Returns:
            dict: Tracked labels and lineage information.
        """
        image_norm = normalize(image)

        cell_tracker = deepcell_tracking.CellTracker(
            image_norm, labels, self.model,
            track_length=self.track_length,
            birth=self.birth, death=self.death,
            division=self.division)

        cell_tracker.track_cells()

        return cell_tracker._track_review_dict()

    def track(self, image, labels, **kwargs):
        """Wrapper around predict() for convenience."""
        return self.predict(image, labels, **kwargs)
