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
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD

from deepcell import losses
from deepcell import image_generators
from deepcell.callbacks import RedirectModel, Evaluate
from deepcell.model_zoo import retinanet_bbox
from deepcell.utils.retinanet_anchor_utils import overlap
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
                       test_size=.1,
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
                       seed=None,
                       **kwargs):
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
        batch_size=batch_size,
        transform=transform,
        transform_kwargs=kwargs,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples)

    val_data = datagen_val.flow(
        test_dict,
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
                     test_size=.1,
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
                     seed=None,
                     **kwargs):
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
        train_data = datagen_val.flow(
            train_dict,
            skip=skip,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs,
            frames_per_batch=frames_per_batch)

        val_data = datagen_val.flow(
            test_dict,
            skip=skip,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs,
            frames_per_batch=frames_per_batch)
    else:
        train_data = datagen.flow(
            train_dict,
            skip=skip,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs)

        val_data = datagen_val.flow(
            test_dict,
            skip=skip,
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
                                 test_size=.1,
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
                                 seed=None,
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

    train_data = datagen.flow(
        train_dict,
        crop_dim=crop_dim,
        batch_size=batch_size,
        min_track_length=min_track_length,
        neighborhood_scale_size=neighborhood_scale_size,
        features=features)

    val_data = datagen_val.flow(
        val_dict,
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
                          backbone,
                          expt='',
                          test_size=.1,
                          n_epoch=10,
                          batch_size=1,
                          num_gpus=None,
                          include_masks=False,
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
                          **kwargs):
    """Train a RetinaNet model from the given backbone

    Adapted from:
        https://github.com/fizyr/keras-retinanet &
        https://github.com/fizyr/keras-maskrcnn
    """
    is_channels_first = K.image_data_format() == 'channels_first'

    if model_name is None:
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = '{}_{}_{}'.format(todays_date, data_name, expt)
    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))

    train_dict, test_dict = get_data(dataset, mode='conv', test_size=test_size)

    n_classes = model.layers[-1].output_shape[1 if is_channels_first else -1]
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

    def regress_loss(y_true, y_pred):
        # separate target and state
        regression = y_pred
        regression_target = y_true[..., :-1]
        anchor_state = y_true[..., -1]

        # filter out "ignore" anchors
        indices = tf.where(K.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute the loss
        loss = losses.smooth_l1(regression_target, regression, sigma=sigma)

        # compute the normalizer: the number of positive anchors
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())

        return K.sum(loss) / normalizer

    def classification_loss(y_true, y_pred):
        # TODO: try weighted_categorical_crossentropy
        labels = y_true[..., :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[..., -1]

        classification = y_pred
        # filter out "ignore" anchors
        indices = tf.where(K.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the loss
        loss = losses.focal(labels, classification, alpha=alpha, gamma=gamma)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return K.sum(loss) / normalizer

    def mask_loss(y_true, y_pred):

        def _mask(y_true, y_pred, iou_threshold=0.5, mask_size=(28, 28)):
            # split up the different predicted blobs
            boxes = y_pred[:, :, :4]
            masks = y_pred[:, :, 4:]

            # split up the different blobs
            annotations = y_true[:, :, :5]
            width = K.cast(y_true[0, 0, 5], dtype='int32')
            height = K.cast(y_true[0, 0, 6], dtype='int32')
            masks_target = y_true[:, :, 7:]

            # reshape the masks back to their original size
            masks_target = K.reshape(masks_target,
                                     (K.shape(masks_target)[0] * K.shape(masks_target)[1],
                                      height, width))
            masks = K.reshape(masks, (K.shape(masks)[0] * K.shape(masks)[1],
                                      mask_size[0], mask_size[1], -1))

            # batch size > 1 fix
            boxes = K.reshape(boxes, (-1, K.shape(boxes)[2]))
            annotations = K.reshape(annotations, (-1, K.shape(annotations)[2]))

            # compute overlap of boxes with annotations
            iou = overlap(boxes, annotations)
            argmax_overlaps_inds = K.argmax(iou, axis=1)
            max_iou = K.max(iou, axis=1)

            # filter those with IoU > 0.5
            indices = tf.where(K.greater_equal(max_iou, iou_threshold))
            boxes = tf.gather_nd(boxes, indices)
            masks = tf.gather_nd(masks, indices)
            argmax_overlaps_inds = tf.gather_nd(argmax_overlaps_inds, indices)
            argmax_overlaps_inds = K.cast(argmax_overlaps_inds, 'int32')
            labels = K.gather(annotations[:, 4], argmax_overlaps_inds)
            labels = K.cast(labels, 'int32')

            # make normalized boxes
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            boxes = K.stack([
                y1 / (K.cast(height, dtype=K.floatx()) - 1),
                x1 / (K.cast(width, dtype=K.floatx()) - 1),
                (y2 - 1) / (K.cast(height, dtype=K.floatx()) - 1),
                (x2 - 1) / (K.cast(width, dtype=K.floatx()) - 1),
            ], axis=1)

            # crop and resize masks_target
            # append a fake channel dimension
            masks_target = K.expand_dims(masks_target, axis=3)
            masks_target = tf.image.crop_and_resize(
                masks_target, boxes, argmax_overlaps_inds, mask_size)

            # remove fake channel dimension
            masks_target = masks_target[:, :, :, 0]

            # gather the predicted masks using the annotation label
            masks = tf.transpose(masks, (0, 3, 1, 2))
            label_indices = K.stack([tf.range(K.shape(labels)[0]), labels], axis=1)
            masks = tf.gather_nd(masks, label_indices)

            # compute mask loss
            mask_loss = K.binary_crossentropy(masks_target, masks)
            normalizer = K.shape(masks)[0] * K.shape(masks)[1] * K.shape(masks)[2]
            normalizer = K.maximum(K.cast(normalizer, K.floatx()), 1)
            mask_loss = K.sum(mask_loss) / normalizer

            return mask_loss

        # if there are no masks annotations, return 0; else, compute the masks loss
        return tf.cond(
            K.any(K.equal(K.shape(y_true), 0)),
            lambda: K.cast_to_floatx(0.0),
            lambda: _mask(y_true, y_pred,
                          iou_threshold=iou_threshold,
                          mask_size=mask_size)
        )

    # evaluation of model is done on `retinanet_bbox`
    if include_masks:
        prediction_model = model
    else:
        prediction_model = retinanet_bbox(
            model, nms=True, class_specific_filter=False)

    loss = {
        'regression': regress_loss,
        'classification': classification_loss
    }

    if include_masks:
        loss['boxes_masks'] = mask_loss

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

    if 'vgg' in backbone or 'densenet' in backbone:
        compute_shapes = make_shapes_callback(model)
    else:
        compute_shapes = guess_shapes

    train_data = datagen.flow(
        train_dict,
        include_masks=include_masks,
        compute_shapes=compute_shapes,
        batch_size=batch_size)

    val_data = datagen_val.flow(
        test_dict,
        include_masks=include_masks,
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
