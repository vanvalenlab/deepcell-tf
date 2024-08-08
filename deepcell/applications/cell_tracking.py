# Copyright 2016-2024 The Van Valen Lab at the California Institute of
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


from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

import deepcell_tracking

from deepcell.applications import Application
from deepcell.utils import fetch_data, extract_archive


@dataclass
class Config:
    inf_key: str
    inf_hash: str
    encoder_key: str
    encoder_hash: str
    distance_threshold: int
    appearance_dim: int
    crop_mode: str
    norm: bool
    birth: float
    death: float
    division: float
    track_length: int
    model_mpp: float


CONFIGS = {
    '1.1': Config(
        inf_hash='ec07d8c0770453e738f8699ceede78e7',
        inf_key='models/NuclearTrackingInf-8.tar.gz',
        encoder_hash='79188d7ae32b40b5bd0ad0f2ac2b53c4',
        encoder_key='models/NuclearTrackingNE-8.tar.gz',
        distance_threshold=64,
        appearance_dim=16,
        crop_mode='fixed',
        norm=True,
        birth=0.99,
        death=0.99,
        division=0.0001,
        track_length=8,
        model_mpp=0.65
    ),
    '1.0': Config(
        inf_hash='5dbd8137be851a0c12557fcde5021444',
        inf_key='models/NuclearTrackingInf-75.tar.gz',
        encoder_hash='a466682c9d1d5e3672325bb8a13ab3e0',
        encoder_key='models/NuclearTrackingNE-75.tar.gz',
        distance_threshold=64,
        appearance_dim=32,
        crop_mode='resize',
        norm=True,
        birth=0.99,
        death=0.99,
        division=0.01,
        track_length=8,
        model_mpp=0.65
    )
}


class CellTracking(Application):
    """Loads a :mod:`deepcell.model_zoo.tracking.GNNTrackingModel` model for
    object tracking with pretrained weights using a simple ``predict`` interface.

    Args:
        model (``tf.keras.model``): Tracking inference model, defaults to latest published model
        neighborhood_encoder (``tf.keras.model``): Tracking neighborhood encoder,
            defaults to latest published model
        distance_threshold (int): Maximum distance between two cells to be considered adjacent
        appearance_dim (int): Length of appearance dimension
        birth (float): Cost of new cell in linear assignment matrix.
        death (float): Cost of cell death in linear assignment matrix.
        division (float): Cost of cell division in linear assignment matrix.
        track_length (int): Number of frames per track
        crop_mode (str): Type of cropping around each cell
        norm (str): Type of normalization layer
    """
    def __init__(self,
                 model,
                 neighborhood_encoder,
                 distance_threshold=8,
                 appearance_dim=32,
                 birth=0.99,
                 death=0.99,
                 division=0.01,
                 track_length=8,
                 embedding_axis=0,
                 crop_mode='resize',
                 norm=True,
                 model_mpp=0.65):
        self.neighborhood_encoder = neighborhood_encoder
        self.distance_threshold = distance_threshold
        self.appearance_dim = appearance_dim
        self.birth = birth
        self.death = death
        self.division = division
        self.track_length = track_length
        self.embedding_axis = embedding_axis
        self.crop_mode = crop_mode
        self.norm = norm

        super().__init__(
            model,
            model_mpp=model_mpp,
            preprocessing_fn=None,
            postprocessing_fn=None)


    @classmethod
    def from_version(cls, version='1.1'):
        """Load a specified version of the model

        1.1: Updates to the Caliban model released in July 2024
        1.0: Original Caliban model released with the September 2023 preprint

        Args:
            version (:obj:`str`, optional): Defaults to '1.1'.
        """
        if version not in CONFIGS:
            raise ValueError(f'Selected version {version} is not available. '
                             f'Choose from {CONFIGS.keys()}')

        config = CONFIGS[version]

        cache_subdir = "models"
        model_dir = Path.home() / ".deepcell" / "models"

        # Load encoder
        archive_path = fetch_data(
            asset_key=config.encoder_key,
            cache_subdir=cache_subdir,
            file_hash=config.encoder_hash
        )
        extract_archive(archive_path, model_dir)
        model_path = model_dir / 'NuclearTrackingNE'
        neighborhood_encoder = tf.keras.models.load_model(model_path)

        # Load inference
        archive_path = fetch_data(
            asset_key=config.inf_key,
            cache_subdir=cache_subdir,
            file_hash=config.inf_hash
        )
        extract_archive(archive_path, model_dir)
        model_path = model_dir / 'NuclearTrackingInf'
        inference = tf.keras.models.load_model(model_path)

        return cls(
            inference,
            neighborhood_encoder,
            distance_threshold=config.distance_threshold,
            appearance_dim=config.appearance_dim,
            birth=config.birth,
            death=config.death,
            division=config.division,
            track_length=config.track_length,
            embedding_axis=0,
            crop_mode=config.crop_mode,
            norm=config.norm,
            model_mpp=config.model_mpp
        )


    def predict(self, image, labels, **kwargs):
        """Using both raw image data and segmentation masks,
        track objects across all frames.

        Args:
            image (numpy.array): Raw image data.
            labels (numpy.array): Labels for ``image``, integer masks.

        Returns:
            dict: Tracked labels and lineage information.
        """

        cell_tracker = deepcell_tracking.CellTracker(
            image,
            labels,
            self.model,
            neighborhood_encoder=self.neighborhood_encoder,
            distance_threshold=self.distance_threshold,
            appearance_dim=self.appearance_dim,
            track_length=self.track_length,
            embedding_axis=self.embedding_axis,
            birth=self.birth,
            death=self.death,
            division=self.division,
            crop_mode=self.crop_mode,
            norm=self.norm)

        cell_tracker.track_cells()

        return cell_tracker._track_review_dict()

    def track(self, image, labels, **kwargs):
        """Wrapper around predict() for convenience."""
        return self.predict(image, labels, **kwargs)
