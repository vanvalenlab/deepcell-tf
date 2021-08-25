# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
              'saved-models/TrackingModel-4.tar.gz')

ENCODER_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'saved-models/TrackingModelNE-2.tar.gz')


class CellTracking(Application):
    """Loads a :mod:`deepcell.model_zoo.tracking.GNNTrackingModel` model for
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
        'batch_size': 4,
        'track_length': 8,
        'lr': 1e-3,
        'clipnorm': 0.001,
        'n_epochs': 30,
        'training_steps_per_epoch': 512,
        'validation_steps': 100,
        'min_lr': 1e-7,
        'training_seed': None,
        'crop_dim': 32
    }

    def __init__(self,
                 model=None,
                 neighborhood_encoder=None,
                 distance_threshold=64,
                 birth=0.99,
                 death=0.99,
                 division=0.9,
                 track_length=8,
                 embedding_axis=0):
        self.neighborhood_encoder = neighborhood_encoder
        self.distance_threshold = distance_threshold
        self.birth = birth
        self.death = death
        self.division = division
        self.track_length = track_length
        self.embedding_axis = embedding_axis

        if self.neighborhood_encoder is None:
            archive_path = tf.keras.utils.get_file(
                'TrackingModelNE.tgz', ENCODER_PATH,
                file_hash='80217fdd8477f0cb827fe72e8ace6542',
                extract=True, cache_subdir='models')
            model_path = os.path.splitext(archive_path)[0]
            self.neighborhood_encoder = tf.keras.models.load_model(model_path)

        if model is None:
            archive_path = tf.keras.utils.get_file(
                'TrackingModelInf.tgz', MODEL_PATH,
                file_hash='2a4d08eb610999563f6a8f06692f8783',
                extract=True, cache_subdir='models')
            model_path = os.path.splitext(archive_path)[0]
            model = tf.keras.models.load_model(model_path)

        super(CellTracking, self).__init__(
            model,
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
            neighborhood_encoder=self.neighborhood_encoder,
            distance_threshold=self.distance_threshold,
            track_length=self.track_length,
            embedding_axis=self.embedding_axis,
            birth=self.birth, death=self.death,
            division=self.division)

        cell_tracker.track_cells()

        return cell_tracker._track_review_dict()

    def track(self, image, labels, **kwargs):
        """Wrapper around predict() for convenience."""
        return self.predict(image, labels, **kwargs)
