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
"""Functions for training convolutional neural networks"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import os

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD

from deepcell import losses
from deepcell import image_generators
from deepcell.callbacks import RedirectModel, Evaluate
from deepcell.model_zoo import retinanet_bbox
from deepcell.utils.retinanet_anchor_utils import make_shapes_callback
from deepcell.utils.retinanet_anchor_utils import guess_shapes
from deepcell.utils.retinanet_anchor_utils import evaluate
from deepcell.utils import train_utils
from deepcell.utils import tracking_utils
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler


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

    if num_gpus >= 2:
        batch_size = batch_size * num_gpus
        model = train_utils.MultiGpuModel(model, num_gpus)

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

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=[
            callbacks.LearningRateScheduler(lr_sched),
            callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=num_gpus >= 2),
            callbacks.TensorBoard(log_dir=os.path.join(log_dir, model_name))
        ])

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
        if focal:
            return losses.weighted_focal_loss(
                y_true, y_pred, gamma=gamma, n_classes=n_classes)
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    if num_gpus >= 2:
        batch_size = batch_size * num_gpus
        model = train_utils.MultiGpuModel(model, num_gpus)

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

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=[
            callbacks.LearningRateScheduler(lr_sched),
            callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=num_gpus >= 2),
            callbacks.TensorBoard(log_dir=os.path.join(log_dir, model_name))
        ])

    model.save_weights(model_path)
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

    if num_gpus >= 2:
        batch_size = batch_size * num_gpus
        model = train_utils.MultiGpuModel(model, num_gpus)

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

    total_train_pairs = tracking_utils.count_pairs(train_dict['y'], same_probability=5.0)
    total_test_pairs = tracking_utils.count_pairs(val_dict['y'], same_probability=5.0)

    # total_train_pairs = tracking_utils.count_pairs(train_dict['y'], same_probability=0.5)
    # total_test_pairs = tracking_utils.count_pairs(val_dict['y'], same_probability=0.5)

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

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=total_train_pairs // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=total_test_pairs // batch_size,
        callbacks=[
            callbacks.LearningRateScheduler(lr_sched),
            callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=num_gpus >= 2),
            callbacks.TensorBoard(log_dir=os.path.join(log_dir, model_name))
        ])

    model.save_weights(model_path)
    np.savez(loss_path, loss_history=loss_history.history)

    return model


def train_model_retinanet(model,
                          dataset,
                          expt='',
                          test_size=.2,
                          n_epoch=10,
                          batch_size=1,
                          num_gpus=None,
                          include_masks=False,
                          panoptic=False,
                          panoptic_weight=0.1,
                          transforms=['watershed'],
                          transforms_kwargs={},
                          anchor_params=None,
                          pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                          min_objects=3,
                          mask_size=(28, 28),
                          optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                          log_dir='/data/tensorboard_logs',
                          model_dir='/data/models',
                          model_name=None,
                          sigma=3.0,
                          alpha=0.25,
                          gamma=2.0,
                          score_threshold=0.01,
                          iou_threshold=0.5,
                          max_detections=100,
                          weighted_average=True,
                          lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                          rotation_range=0,
                          flip=True,
                          shear=0,
                          zoom_range=0,
                          compute_map=True,
                          seed=0,
                          **kwargs):
    """Train a RetinaNet model from the given backbone.

    Adapted from:
        https://github.com/fizyr/keras-retinanet &
        https://github.com/fizyr/keras-maskrcnn

    Args:
        model (tensorflow.keras.Model): The model to train.
        dataset (str): Path to a dataset to train the model with.
        expt (str): Experiment, substring to include in model name.
        test_size (float): Percent of data to leave as test data.
        n_epoch (int): Number of training epochs.
        batch_size (int): Number of batches per training step.
        num_gpus (int): The number of GPUs to train on.
        include_masks (bool): Whether to generate masks using MaskRCNN.
        panoptic (bool): Whether to include semantic segmentation heads.
        panoptic_weight (float): Weight applied to the semantic loss.
        transforms (list): List of transform names as strings. Each transform
            will have its own semantic segmentation head.
        transforms_kwargs (list): List of dicts of optional values for each
            transform in transforms.
        anchor_params (AnchorParameters): Struct containing anchor parameters.
            If None, default values are used.
        pyramid_levels (list): Pyramid levels to attach
            the object detection heads to.
        min_objects (int): If a training image has fewer than min_objects
            objects, the image will not be used for training.
        mask_size (tuple): The size of the masks.
        log_dir (str): Filepath to save tensorboard logs. If None, disables
            the tensorboard callback.
        model_dir (str): Directory to save the model file.
        model_name (str): Name of the model (and name of output file).
        sigma (float): The point where the loss changes from L2 to L1.
        alpha (float): Scale the focal weight with alpha.
        gamma (float): Take the power of the focal weight with gamma.
        iou_threshold (float): The threshold used to consider when a detection
            is positive or negative.
        score_threshold (float): The score confidence threshold
            to use for detections.
        max_detections (int): The maximum number of detections to use per image
        weighted_average (bool): Use a weighted average in evaluation.
        optimizer (object): Pre-initialized optimizer object (SGD, Adam, etc.)
        lr_sched (function): Learning rate schedular function
        rotation_range (int): Maximum rotation range for image augmentation
        flip (bool): Enables horizontal and vertical flipping for augmentation
        shear (int): Maximum rotation range for image augmentation
        zoom_range (tuple): Minimum and maximum zoom values (0.8, 1.2)
        seed (int): Random seed
        compute_map (bool): Whether to compute mAP at end of training.
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

    train_dict, test_dict = get_data(dataset, seed=seed, test_size=test_size)

    channel_axis = 1 if is_channels_first else -1
    n_classes = model.layers[-1].output_shape[channel_axis]

    if panoptic:
        n_semantic_classes = [layer.output_shape[channel_axis]
                              for layer in model.layers if 'semantic' in layer.name]
    else:
        n_semantic_classes = []

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    if num_gpus >= 1e6:
        batch_size = batch_size * num_gpus
        model = train_utils.MultiGpuModel(model, num_gpus)

    print('Training on {} GPUs'.format(num_gpus))

    # evaluation of model is done on `retinanet_bbox`
    if include_masks:
        prediction_model = model
    else:
        prediction_model = retinanet_bbox(
            model,
            nms=True,
            anchor_params=anchor_params,
            num_semantic_heads=len(n_semantic_classes),
            panoptic=panoptic,
            class_specific_filter=False)

    retinanet_losses = losses.RetinaNetLosses(sigma=sigma, alpha=alpha, gamma=gamma,
                                              iou_threshold=iou_threshold,
                                              mask_size=mask_size)

    def semantic_loss(n_classes):
        def _semantic_loss(y_pred, y_true):
            return panoptic_weight * losses.weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes)
        return _semantic_loss

    loss = {
        'regression': retinanet_losses.regress_loss,
        'classification': retinanet_losses.classification_loss
    }

    if include_masks:
        loss['masks'] = retinanet_losses.mask_loss

    if panoptic:
        # Give losses for all of the semantic heads
        for layer in model.layers:
            if 'semantic' in layer.name:
                n_classes = layer.output_shape[channel_axis]
                loss[layer.name] = semantic_loss(n_classes)

    model.compile(loss=loss, optimizer=optimizer)

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
    datagen = image_generators.RetinaNetGenerator(
        # fill_mode='constant',  # for rotations
        rotation_range=rotation_range,
        shear_range=shear,
        zoom_range=zoom_range,
        horizontal_flip=flip,
        vertical_flip=flip)

    datagen_val = image_generators.RetinaNetGenerator(
        # fill_mode='constant',  # for rotations
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    # if 'vgg' in backbone or 'densenet' in backbone:
    #     compute_shapes = make_shapes_callback(model)
    # else:
    #     compute_shapes = guess_shapes

    compute_shapes = guess_shapes

    train_data = datagen.flow(
        train_dict,
        seed=seed,
        include_masks=include_masks,
        panoptic=panoptic,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        pyramid_levels=pyramid_levels,
        min_objects=min_objects,
        anchor_params=anchor_params,
        compute_shapes=compute_shapes,
        batch_size=batch_size)

    val_data = datagen_val.flow(
        test_dict,
        seed=seed,
        include_masks=include_masks,
        panoptic=panoptic,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        pyramid_levels=pyramid_levels,
        min_objects=min_objects,
        anchor_params=anchor_params,
        compute_shapes=compute_shapes,
        batch_size=batch_size)

    tensorboard_callback = callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, model_name))

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=[
            callbacks.LearningRateScheduler(lr_sched),
            callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=num_gpus >= 2),
            tensorboard_callback,
            callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.1,
                patience=10, verbose=1,
                mode='auto', min_delta=0.0001,
                cooldown=0, min_lr=0),
            RedirectModel(
                Evaluate(val_data,
                         iou_threshold=iou_threshold,
                         score_threshold=score_threshold,
                         max_detections=max_detections,
                         tensorboard=tensorboard_callback,
                         weighted_average=weighted_average),
                prediction_model),
        ])

    model.save_weights(model_path)
    np.savez(loss_path, loss_history=loss_history.history)

    if compute_map:
        average_precisions = evaluate(
            val_data,
            prediction_model,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )

        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  label, 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
        else:
            print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
                sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
            print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

    return model
