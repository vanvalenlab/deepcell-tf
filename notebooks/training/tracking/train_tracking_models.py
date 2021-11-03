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
"""Custom Layers"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import wandb
import errno
import deepcell

import tensorflow as tf
import numpy as np

from deepcell.data.tracking import prepare_dataset
from deepcell.data.tracking import Track, concat_tracks
from deepcell.model_zoo.tracking import GNNTrackingModel
from deepcell.utils.tracking_utils import trks_stats, load_trks

from tensorflow_addons.optimizers import RectifiedAdam as RAdam

# from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import CSVLogger

# Verify GPU count
from deepcell import train_utils
num_gpus = train_utils.count_gpus()
print('Training on {} GPUs'.format(num_gpus))


# setup directories
ROOT_DIR = '/data/tracking_data'  # TODO: Change this! Usually a mounted volume

MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'models'))
LOG_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'logs'))
DATA_DIR = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
OUTPUT_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'nuc_tracking'))

# create directories if they do not exist
for d in (MODEL_DIR, LOG_DIR, OUTPUT_DIR):
    try:
        os.makedirs(d)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

print()
print('loading tracking data')
print()

# Load and view stats on this file
filename = 'train.trks'
path = os.path.join('../trk_data/',filename)
trks_data = load_trks(path)

dataset_sizes = os.path.abspath(os.path.join(ROOT_DIR, 'dataset_idxs_dvc.npy'))
dataset_indicies = np.load(dataset_sizes, allow_pickle=True).tolist()

# from deepcell.datasets.tracked import hek293
# filename = 'sample_tracking.trks'
# (X_train, y_train), (X_test, y_test) = hek293.load_tracked_data(filename)
# path = os.path.join(DATA_DIR, filename)
# trks_data = load_trks(path)
# all_tracks = Track(tracked_data=trks_data)
# track_info = concat_tracks([all_tracks])

"""
Functions for metrics
"""
def filter_and_flatten(y_true, y_pred):
    n_classes = tf.shape(y_true)[-1]
    new_shape = [-1, n_classes]
    y_true = tf.reshape(y_true, new_shape)
    y_pred = tf.reshape(y_pred, new_shape)

    # Mask out the padded cells
    y_true_reduced = tf.reduce_sum(y_true, axis=-1)
    good_loc = tf.where(y_true_reduced == 1)[:, 0]

    y_true = tf.gather(y_true, good_loc, axis=0)
    y_pred = tf.gather(y_pred, good_loc, axis=0)
    return y_true, y_pred


class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = filter_and_flatten(y_true, y_pred)
        super(Recall, self).update_state(y_true, y_pred, sample_weight)


class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = filter_and_flatten(y_true, y_pred)
        super(Precision, self).update_state(y_true, y_pred, sample_weight)


def loss_function(y_true, y_pred):
    y_true, y_pred = filter_and_flatten(y_true, y_pred)
    return deepcell.losses.weighted_categorical_crossentropy(
        y_true, y_pred,
        n_classes=tf.shape(y_true)[-1],
        axis=-1)

# Define optimizer
optimizer = RAdam(learning_rate=1e-3, clipnorm=0.001)

# Define the loss function
losses = {'temporal_adj_matrices': loss_function}

# Define metrics
metrics = [
    Recall(class_id=0, name='same_recall'),
    Recall(class_id=1, name='different_recall'),
    Recall(class_id=2, name='daughter_recall'),
    Precision(class_id=0, name='same_precision'),
    Precision(class_id=1, name='different_precision'),
    Precision(class_id=2, name='daughter_precision'),
]


"""
Set up training parameters
"""
seed = 1   # random seed for training/validation data split
batch_size = 1
track_length = 8  # only train on 8 frames at once
val_size = .20  # % of data saved as validation
test_size = .1  # % of data held out as a test set
n_epochs = 8  # number of training epochs

steps_per_epoch = 512
validation_steps = 100


translation_range = 512 #X_train.shape[-2]

n_layers = 1


for i in range(len(dataset_indicies)):
    new_data = {}
    new_data['lineages'] = list(np.array(trks_data['lineages'])[dataset_indicies[i]])
    new_data['X'] = trks_data['X'][dataset_indicies[i],...]
    new_data['y'] = trks_data['y'][dataset_indicies[i],...]

    ds_size = len(dataset_indicies[i])

    print()
    print('data idx', i, 'size', ds_size)
    print()

    all_tracks = Track(tracked_data=new_data)
    track_info = concat_tracks([all_tracks])

    # find maximum number of cells in any frame
    max_cells = track_info['appearances'].shape[2]

    train_data, val_data, test_data = prepare_dataset(
        track_info,
        rotation_range=180,
        translation_range=translation_range,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        batch_size=batch_size,
        track_length=track_length)

    graph_layers = ['gcn', 'gcs', 'se2c', 'se2t']

    for layer in graph_layers:

        print()
        print('graph layer', layer)
        print()

        model_name = 'graph_tracking_model_seed{}'.format(seed)
        model_path = os.path.join(MODEL_DIR, model_name)

        train_log = os.path.join(ROOT_DIR, f'train_logs/training_log_{layer}_{ds_size}.csv')
        csv_logger = CSVLogger(train_log)

        # run = wandb.init(project='cell_tracking', reinit=True)
        # wandb.run.name = layer+f'_datasize_{ds_size}'
        # wandb.log({'metrics': metrics,
        #            'losses': losses})

        tm = GNNTrackingModel(max_cells=max_cells, n_layers=n_layers, graph_layer=layer)

        # Compile model
        tm.training_model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        
        # Train the model
        train_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor='val_loss',
                save_best_only=True, verbose=1,
                save_weights_only=False),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, verbose=1,
                patience=3, min_lr=1e-7), csv_logger]

        loss_history = tm.training_model.fit(
            train_data,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            validation_steps=validation_steps,
            epochs=n_epochs,
            verbose=1,
            callbacks=train_callbacks)

        # Save models for prediction
        inf_path = os.path.join(MODEL_DIR, f'TrackingModelInf_{layer}_datasize_{ds_size}')
        ne_path = os.path.join(MODEL_DIR, f'TrackingModelNE_{layer}_datasize_{ds_size}')

        tm.inference_model.save(inf_path)
        tm.neighborhood_encoder.save(ne_path)

        # run.finish()