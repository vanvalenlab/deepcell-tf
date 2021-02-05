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
"""Functions for training convolutional neural networks"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MSE
from tensorflow.python.data import Dataset

from deepcell import losses
from deepcell import image_generators
from deepcell.utils import train_utils
from deepcell.utils import tracking_utils
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler
from deepcell.utils.train_utils import get_callbacks


def train_model_sample(model,
                       dataset,
                       expt='',
                       test_size=.2,
                       n_epoch=10,
                       batch_size=32,
                       num_gpus=None,
                       transform=None,
                       window_size=None,
                       balance_classes=True,
                       max_class_samples=None,
                       log_dir='/data/tensorboard_logs',
                       model_dir='/data/models',
                       model_name=None,
                       focal=False,
                       gamma=0.5,
                       optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                       lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                       rotation_range=0,
                       flip=False,
                       shear=0,
                       zoom_range=0,
                       seed=0,
                       **kwargs):
    """Train a model using sample mode.

    Args:
        model (tensorflow.keras.Model): The model to train.
        dataset (str): Path to a dataset to train the model with.
        expt (str): Experiment, substring to include in model name.
        test_size (float): Percent of data to leave as test data.
        n_epoch (int): Number of training epochs.
        batch_size (int): Number of batches per training step.
        num_gpus (int): The number of GPUs to train on.
        transform (str): Defines the transformation of the training data.
            One of 'watershed', 'fgbg', 'pixelwise'.
        window_size (tuple(int, int)): Size of sampling window
        balance_classes (bool): Whether to perform class-balancing on data
        max_class_samples (int): Maximum number of examples per class to sample
        log_dir (str): Filepath to save tensorboard logs. If None, disables
            the tensorboard callback.
        model_dir (str): Directory to save the model file.
        model_name (str): Name of the model (and name of output file).
        focal (bool): If true, uses focal loss.
        gamma (float): Parameter for focal loss
        optimizer (object): Pre-initialized optimizer object (SGD, Adam, etc.)
        lr_sched (function): Learning rate schedular function
        rotation_range (int): Maximum rotation range for image augmentation
        flip (bool): Enables horizontal and vertical flipping for augmentation
        shear (int): Maximum rotation range for image augmentation
        zoom_range (tuple): Minimum and maximum zoom values (0.8, 1.2)
        seed (int): Random seed
        kwargs (dict): Other parameters to pass to _transform_masks

    Returns:
        tensorflow.keras.Model: The trained model
    """
    is_channels_first = K.image_data_format() == 'channels_first'

    if model_name is None:
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = '{}_{}_{}'.format(todays_date, data_name, expt)
    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))

    train_dict, test_dict = get_data(dataset, test_size=test_size, seed=seed)

    n_classes = model.layers[-1].output_shape[1 if is_channels_first else -1]

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        if isinstance(transform, str) and transform.lower() == 'disc':
            return losses.discriminative_instance_loss(y_true, y_pred)
        if focal:
            return losses.weighted_focal_loss(
                y_true, y_pred, gamma=gamma, n_classes=n_classes)
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    print('Training on {} GPUs'.format(num_gpus))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    if train_dict['X'].ndim == 4:
        DataGenerator = image_generators.SampleDataGenerator
        window_size = window_size if window_size else (30, 30)
    elif train_dict['X'].ndim == 5:
        DataGenerator = image_generators.SampleMovieDataGenerator
        window_size = window_size if window_size else (30, 30, 3)
    else:
        raise ValueError('Expected `X` to have ndim 4 or 5. Got',
                         train_dict['X'].ndim)

    # this will do preprocessing and realtime data augmentation
    datagen = DataGenerator(
        rotation_range=rotation_range,
        shear_range=shear,
        zoom_range=zoom_range,
        horizontal_flip=flip,
        vertical_flip=flip)

    # no validation augmentation
    datagen_val = DataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    train_data = datagen.flow(
        train_dict,
        seed=seed,
        batch_size=batch_size,
        transform=transform,
        transform_kwargs=kwargs,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples)

    val_data = datagen_val.flow(
        test_dict,
        seed=seed,
        batch_size=batch_size,
        transform=transform,
        transform_kwargs=kwargs,
        window_size=window_size,
        balance_classes=False,
        max_class_samples=max_class_samples)

    train_callbacks = get_callbacks(
        model_path, lr_sched=lr_sched,
        tensorboard_log_dir=log_dir,
        save_weights_only=num_gpus >= 2,
        monitor='val_loss', verbose=1)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=train_callbacks)

    np.savez(loss_path, loss_history=loss_history.history)

    return model


def train_model_conv(model,
                     dataset,
                     expt='',
                     test_size=.2,
                     n_epoch=10,
                     batch_size=1,
                     num_gpus=None,
                     frames_per_batch=5,
                     transform=None,
                     optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                     log_dir='/data/tensorboard_logs',
                     model_dir='/data/models',
                     model_name=None,
                     focal=False,
                     gamma=0.5,
                     lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                     rotation_range=0,
                     flip=True,
                     shear=0,
                     zoom_range=0,
                     seed=0,
                     **kwargs):
    """Train a model using fully convolutional mode.

    Args:
        model (tensorflow.keras.Model): The model to train.
        dataset (str): Path to a dataset to train the model with.
        expt (str): Experiment, substring to include in model name.
        test_size (float): Percent of data to leave as test data.
        n_epoch (int): Number of training epochs.
        batch_size (int): Number of batches per training step.
        num_gpus (int): The number of GPUs to train on.
        frames_per_batch (int): Number of training frames if training 3D data.
        transform (str): Defines the transformation of the training data.
            One of 'watershed', 'fgbg', 'pixelwise'.
        log_dir (str): Filepath to save tensorboard logs. If None, disables
            the tensorboard callback.
        model_dir (str): Directory to save the model file.
        model_name (str): Name of the model (and name of output file).
        focal (bool): If true, uses focal loss.
        gamma (float): Parameter for focal loss
        optimizer (object): Pre-initialized optimizer object (SGD, Adam, etc.)
        lr_sched (function): Learning rate schedular function
        rotation_range (int): Maximum rotation range for image augmentation
        flip (bool): Enables horizontal and vertical flipping for augmentation
        shear (int): Maximum rotation range for image augmentation
        zoom_range (tuple): Minimum and maximum zoom values (0.8, 1.2)
        seed (int): Random seed
        kwargs (dict): Other parameters to pass to _transform_masks

    Returns:
        tensorflow.keras.Model: The trained model
    """
    is_channels_first = K.image_data_format() == 'channels_first'

    if model_name is None:
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = '{}_{}_{}'.format(todays_date, data_name, expt)
    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))

    train_dict, test_dict = get_data(dataset, test_size=test_size, seed=seed)

    n_classes = model.layers[-1].output_shape[1 if is_channels_first else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        if isinstance(transform, str) and transform.lower() == 'disc':
            return losses.discriminative_instance_loss(y_true, y_pred)
        if isinstance(transform, str) and transform.lower() == 'watershed-cont':
            return MSE(y_true, y_pred)
        if focal:
            return losses.weighted_focal_loss(
                y_true, y_pred, gamma=gamma, n_classes=n_classes)
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    print('Training on {} GPUs'.format(num_gpus))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    if isinstance(model.output_shape, list):
        skip = len(model.output_shape) - 1
    else:
        skip = None

    if train_dict['X'].ndim == 4:
        DataGenerator = image_generators.ImageFullyConvDataGenerator
    elif train_dict['X'].ndim == 5:
        DataGenerator = image_generators.MovieDataGenerator
    else:
        raise ValueError('Expected `X` to have ndim 4 or 5. Got',
                         train_dict['X'].ndim)

    if num_gpus >= 2:
        # Each GPU must have at least one validation example
        if test_dict['y'].shape[0] < num_gpus:
            raise ValueError('Not enough validation data for {} GPUs. '
                             'Received {} validation sample.'.format(
                                 test_dict['y'].shape[0], num_gpus))

        # When using multiple GPUs and skip_connections,
        # the training data must be evenly distributed across all GPUs
        num_train = train_dict['y'].shape[0]
        nb_samples = num_train - num_train % batch_size
        if nb_samples:
            train_dict['y'] = train_dict['y'][:nb_samples]
            train_dict['X'] = train_dict['X'][:nb_samples]

    # this will do preprocessing and realtime data augmentation
    datagen = DataGenerator(
        rotation_range=rotation_range,
        shear_range=shear,
        zoom_range=zoom_range,
        horizontal_flip=flip,
        vertical_flip=flip)

    datagen_val = DataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    if train_dict['X'].ndim == 5:
        train_data = datagen.flow(
            train_dict,
            skip=skip,
            seed=seed,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs,
            frames_per_batch=frames_per_batch)

        val_data = datagen_val.flow(
            test_dict,
            skip=skip,
            seed=seed,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs,
            frames_per_batch=frames_per_batch)
    else:
        train_data = datagen.flow(
            train_dict,
            skip=skip,
            seed=seed,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs)

        val_data = datagen_val.flow(
            test_dict,
            skip=skip,
            seed=seed,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs)

    train_callbacks = get_callbacks(
        model_path, lr_sched=lr_sched,
        tensorboard_log_dir=log_dir,
        save_weights_only=num_gpus >= 2,
        monitor='val_loss', verbose=1)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=train_callbacks)

    np.savez(loss_path, loss_history=loss_history.history)

    return model


def train_model_siamese_daughter(model,
                                 dataset,
                                 expt='',
                                 test_size=.2,
                                 n_epoch=100,
                                 batch_size=1,
                                 num_gpus=None,
                                 crop_dim=32,
                                 min_track_length=1,
                                 neighborhood_scale_size=10,
                                 features=None,
                                 optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                                 log_dir='/data/tensorboard_logs',
                                 model_dir='/data/models',
                                 model_name=None,
                                 focal=False,
                                 gamma=0.5,
                                 lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                                 rotation_range=0,
                                 flip=True,
                                 shear=0,
                                 zoom_range=0,
                                 seed=0,
                                 **kwargs):
    is_channels_first = K.image_data_format() == 'channels_first'

    if model_name is None:
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = '{}_{}_[{}]_neighs={}_epochs={}_seed={}_{}'.format(
            todays_date, data_name, ','.join(f[0] for f in sorted(features)),
            neighborhood_scale_size, n_epoch, seed, expt)
    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))

    print('training on dataset:', dataset)
    print('saving model at:', model_path)
    print('saving loss at:', loss_path)

    train_dict, val_dict = get_data(dataset, mode='siamese_daughters',
                                    seed=seed, test_size=test_size)

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', val_dict['X'].shape)
    print('y_test shape:', val_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)

    n_classes = model.layers[-1].output_shape[1 if is_channels_first else -1]

    def loss_function(y_true, y_pred):
        if focal:
            return losses.weighted_focal_loss(y_true, y_pred,
                                              gamma=gamma,
                                              n_classes=n_classes,
                                              from_logits=False)
        return losses.weighted_categorical_crossentropy(y_true, y_pred,
                                                        n_classes=n_classes,
                                                        from_logits=False)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    print('Training on {} GPUs'.format(num_gpus))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = image_generators.SiameseDataGenerator(
        rotation_range=rotation_range,
        shear_range=shear,
        zoom_range=zoom_range,
        horizontal_flip=flip,
        vertical_flip=flip)

    datagen_val = image_generators.SiameseDataGenerator(
        rotation_range=0,
        zoom_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    # same_probability values have varied from 0.5 to 5.0
    total_train_pairs = tracking_utils.count_pairs(train_dict['y'], same_probability=5.0)
    total_test_pairs = tracking_utils.count_pairs(val_dict['y'], same_probability=5.0)

    train_data = datagen.flow(
        train_dict,
        seed=seed,
        crop_dim=crop_dim,
        batch_size=batch_size,
        min_track_length=min_track_length,
        neighborhood_scale_size=neighborhood_scale_size,
        features=features)

    val_data = datagen_val.flow(
        val_dict,
        seed=seed,
        crop_dim=crop_dim,
        batch_size=batch_size,
        min_track_length=min_track_length,
        neighborhood_scale_size=neighborhood_scale_size,
        features=features)

    print('total_train_pairs:', total_train_pairs)
    print('total_test_pairs:', total_test_pairs)
    print('batch size:', batch_size)
    print('validation_steps: ', total_test_pairs // batch_size)

    # Make dicts to map the two generator outputs to the Dataset and model
    # input here is model input and output is model output
    features = sorted(features)

    input_type_dict = {}
    input_shape_dict = {}
    for feature in features:

        feature_name1 = '{}_input1'.format(feature)
        feature_name2 = '{}_input2'.format(feature)

        input_type_dict[feature_name1] = tf.float32
        input_type_dict[feature_name2] = tf.float32

        if feature == 'appearance':
            app1 = tuple([None, train_data.min_track_length,
                          train_data.crop_dim, train_data.crop_dim, 1])
            app2 = tuple([None, 1, train_data.crop_dim, train_data.crop_dim, 1])

            input_shape_dict[feature_name1] = app1
            input_shape_dict[feature_name2] = app2

        elif feature == 'distance':
            dist1 = tuple([None, train_data.min_track_length, 2])
            dist2 = tuple([None, 1, 2])

            input_shape_dict[feature_name1] = dist1
            input_shape_dict[feature_name2] = dist2

        elif feature == 'neighborhood':
            neighborhood_size = 2 * train_data.neighborhood_scale_size + 1
            neigh1 = tuple([None, train_data.min_track_length,
                            neighborhood_size, neighborhood_size, 1])
            neigh2 = tuple([None, 1, neighborhood_size, neighborhood_size, 1])

            input_shape_dict[feature_name1] = neigh1
            input_shape_dict[feature_name2] = neigh2

        elif feature == 'regionprop':
            rprop1 = tuple([None, train_data.min_track_length, 3])
            rprop2 = tuple([None, 1, 3])

            input_shape_dict[feature_name1] = rprop1
            input_shape_dict[feature_name2] = rprop2

    output_type_dict = {'classification': tf.int32}
    # Ouput_shape has to be None because we dont know how many cells
    output_shape_dict = {'classification': (None, 3)}

    train_dataset = Dataset.from_generator(
        lambda: train_data,
        (input_type_dict, output_type_dict),
        output_shapes=(input_shape_dict, output_shape_dict))
    val_dataset = Dataset.from_generator(
        lambda: val_data,
        (input_type_dict, output_type_dict),
        output_shapes=(input_shape_dict, output_shape_dict))

    train_callbacks = get_callbacks(
        model_path, lr_sched=lr_sched,
        tensorboard_log_dir=log_dir,
        save_weights_only=num_gpus >= 2,
        monitor='val_loss', verbose=1)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit(
        train_dataset,
        steps_per_epoch=total_train_pairs // batch_size,
        epochs=n_epoch,
        validation_data=val_dataset,
        validation_steps=total_test_pairs // batch_size,
        callbacks=train_callbacks)

    np.savez(loss_path, loss_history=loss_history.history)

    return model
