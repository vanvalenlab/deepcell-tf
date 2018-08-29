"""
training.py

Functions for training convolutional neural networks

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.utils import to_categorical as keras_to_categorical

from deepcell import losses
from deepcell import image_generators as generators
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler
from deepcell.utils.transform_utils import to_categorical


CHANNELS_FIRST = K.image_data_format() == 'channels_first'


def train_model_sample(model=None, dataset=None, optimizer=None,
                       expt='', it=0, batch_size=32, n_epoch=100,
                       window_size=(30, 30),
                       balance_classes=True, max_class_samples=None,
                       direc_save='/data/models', direc_data='/data/npz_data',
                       lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                       rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='sample')
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.SampleDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    datagen_val = generators.SampleDataGenerator(
        rotation_range=0,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=0,  # randomly flip images
        vertical_flip=0)  # randomly flip images

    train_data = datagen.flow(
        train_dict,
        batch_size=batch_size,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples)

    val_data = datagen_val.flow(
        test_dict,
        batch_size=batch_size,
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
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    np.savez(file_name_save_loss, loss_history=loss_history.history)


def train_model_conv(model=None, dataset=None, optimizer=None,
                     expt='', it=0, batch_size=1, n_epoch=100,
                     direc_save='/data/models', direc_data='/data/npz_data',
                     lr_sched=rate_scheduler(lr=0.01, decay=0.95), skip=None,
                     rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='conv')

    class_weights = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.ImageFullyConvDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    datagen_val = generators.ImageFullyConvDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, skip=skip),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=datagen_val.flow(test_dict, batch_size=batch_size, skip=skip),
        validation_steps=test_dict['y'].shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_siamese(model=None, dataset=None, optimizer=None,
                        expt='', it=0, batch_size=1, n_epoch=100,
                        direc_save='/data/models', direc_data='/data/npz_data',
                        lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                        rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='siamese')

    class_weights = train_dict['class_weights']
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)

    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]

    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.SiameseDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    datagen_val = generators.SiameseDataGenerator(
        rotation_range=0,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=0,  # randomly flip images
        vertical_flip=0)  # randomly flip images

    def count_pairs(y):
        """
        Compute number of training samples needed to (stastically speaking)
        observe all cell pairs.
        Assume that the number of images is encoded in the second dimension.
        Assume that y values are a cell-uniquely-labeled mask.
        Assume that a cell is paired with one of its other frames 50% of the time
        and a frame from another cell 50% of the time.
        """
        # TODO: channels_first axes
        total_pairs = 0
        for image_set in range(y.shape[0]):
            set_cells = 0
            cells_per_image = []
            for image in range(y.shape[1]):
                image_cells = int(y[image_set, image, :, :, :].max())
                set_cells = set_cells + image_cells
                cells_per_image.append(image_cells)

            # Since there are many more possible non-self pairings than there are self pairings,
            # we want to estimate the number of possible non-self pairings and then multiply
            # that number by two, since the odds of getting a non-self pairing are 50%, to
            # find out how many pairs we would need to sample to (statistically speaking)
            # observe all possible cell-frame pairs.
            # We're going to assume that the average cell is present in every frame. This will
            # lead to an underestimate of the number of possible non-self pairings, but it's
            # unclear how significant the underestimate is.
            average_cells_per_frame = int(sum(cells_per_image) / len(cells_per_image))
            non_self_cellframes = (average_cells_per_frame - 1) * len(cells_per_image)
            non_self_pairings = non_self_cellframes * max(cells_per_image)
            cell_pairings = non_self_pairings * 2
            total_pairs = total_pairs + cell_pairings
        return total_pairs

    # This shouldn't remain long term.
    magic_number = 2048  # A power of 2 chosen just to reduce training time.
    total_train_pairs = count_pairs(train_dict['y'])
    total_train_pairs = int(total_train_pairs // magic_number)

    total_test_pairs = count_pairs(test_dict['y'])
    total_test_pairs = int(total_test_pairs // magic_number)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size),
        steps_per_epoch=total_train_pairs // batch_size,
        epochs=n_epoch,
        validation_data=datagen_val.flow(test_dict, batch_size=batch_size),
        validation_steps=total_test_pairs // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_watershed(model=None, dataset=None, optimizer=None, erosion_width=None,
                          expt='', it=0, batch_size=1, n_epoch=100, distance_bins=16,
                          direc_save='/data/models', direc_data='/data/npz_data',
                          lr_sched=rate_scheduler(lr=0.01, decay=0.95), skip=None,
                          rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='conv')

    # if class_weight is None:
    #     class_weight = train_dict['class_weights']

    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        # TODO: implement direction loss
        pass

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.WatershedDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # no augmentation for validation data
    datagen_val = generators.WatershedDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, distance_bins=distance_bins, erosion_width=erosion_width, skip=skip),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        class_weight=class_weight,
        validation_data=datagen_val.flow(test_dict, batch_size=batch_size, distance_bins=distance_bins, erosion_width=erosion_width, skip=skip),
        validation_steps=test_dict['X'].shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_watershed_sample(model=None, dataset=None, optimizer=None,
                                 window_size=(30, 30), balance_classes=True,
                                 max_class_samples=None, erosion_width=None,
                                 distance_bins=16, expt='', it=0, batch_size=1, n_epoch=100,
                                 direc_save='/data/models', direc_data='/data/npz_data',
                                 lr_sched=rate_scheduler(lr=0.01, decay=0.95), skip=None,
                                 rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='conv')

    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.WatershedSampleDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # no augmentation for validation data
    datagen_val = generators.WatershedSampleDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    train_data = datagen.flow(
        train_dict,
        batch_size=batch_size,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples,
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        skip=skip)

    val_data = datagen_val.flow(
        train_dict,
        batch_size=batch_size,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples,
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        skip=skip)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        class_weight=class_weight,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_disc(model=None, dataset=None, optimizer=None,
                     expt='', it=0, batch_size=1, n_epoch=100,
                     direc_save='/data/models', direc_data='/data/npz_data',
                     lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                     rotation_range=0, flip=True, shear=0, class_weight=None,
                     zoom_range=[0.5, 2]):
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
    training_data_file_name = os.path.join(direc_data, dataset + '.npz')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='conv')

    class_weights = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return losses.discriminative_instance_loss(y_true, y_pred)

    model.compile(loss=loss_function, optimizer=optimizer)

    datagen = generators.DiscDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip,  # randomly flip images
        zoom_range=zoom_range)  # randomly zoom in/out images

    datagen_test = generators.DiscDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=False,
        vertical_flip=False)

    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=datagen_test.flow(test_dict, batch_size=batch_size),
        validation_steps=test_dict['X'].shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_disc_3D(model=None, dataset=None, optimizer=None,
                        expt='', it=0, batch_size=1, n_epoch=100,
                        direc_save='/data/models', direc_data='/data/npz_data',
                        lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                        frames_per_batch=10, rotation_range=0, flip=True,
                        shear=0, class_weight=None, zoom_range=[0.5, 2]):
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
    training_data_file_name = os.path.join(direc_data, dataset + '.npz')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='conv')

    class_weights = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return losses.discriminative_instance_loss(y_true, y_pred)

    model.compile(loss=loss_function, optimizer=optimizer)

    datagen = generators.DiscMovieDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip,  # randomly flip images
        zoom_range=zoom_range)  # randomly zoom in/out images

    datagen_test = generators.DiscMovieDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=False,
        vertical_flip=False)

    time_axis = 2 if CHANNELS_FIRST else 1
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, frames_per_batch=frames_per_batch),
        steps_per_epoch=(train_dict['y'].shape[0] * train_dict['y'].shape[time_axis] // frames_per_batch) // batch_size,
        epochs=n_epoch,
        validation_data=datagen_test.flow(test_dict, batch_size=batch_size, frames_per_batch=frames_per_batch),
        validation_steps=(test_dict['X'].shape[0] * test_dict['X'].shape[time_axis] // frames_per_batch) // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_movie(model=None, dataset=None, optimizer=None,
                      expt='', it=0, batch_size=1, n_epoch=100,
                      direc_save='/data/models', direc_data='/data/npz_data',
                      lr_sched=rate_scheduler(lr=0.01, decay=0.95), skip=None,
                      frames_per_batch=10, rotation_range=0, flip=True,
                      shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, '{}.npz'.format(dataset))
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='movie')
    X_train, y_train = train_dict['X'], train_dict['y']

    class_weights = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.MovieDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # no augmentation for validation data
    datagen_val = generators.MovieDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    # set all cell IDs to 1.  We care about is/is not a cell, not the ID
    train_dict['y'][train_dict['y'] > 0] = 1
    test_dict['y'][test_dict['y'] > 0] = 1

    if class_weights is None:
        class_weights = compute_class_weight(
            'balanced',
            y=train_dict['y'].reshape(train_dict['y'].size),
            classes=np.unique(train_dict['y']))

    # keras to_categorical will not work with channels_first data.
    # if channels_first, convert to channels_last
    if K.image_data_format() == 'channels_first':
        train_dict['y'] = np.rollaxis(train_dict['y'], 1, 5)
        test_dict['y'] = np.rollaxis(test_dict['y'], 1, 5)

    # use to_categorical to one-hot encode each feaeture
    train_dict['y'] = keras_to_categorical(train_dict['y'])
    test_dict['y'] = keras_to_categorical(test_dict['y'])

    # if channels_first, roll axis back to expected shape
    if K.image_data_format() == 'channels_first':
        train_dict['y'] = np.rollaxis(train_dict['y'], 4, 1)
        test_dict['y'] = np.rollaxis(test_dict['y'], 4, 1)

    time_axis = 2 if CHANNELS_FIRST else 1
    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, frames_per_batch=frames_per_batch, skip=skip),
        steps_per_epoch=(train_dict['y'].shape[0] * train_dict['y'].shape[time_axis] // frames_per_batch) // batch_size,
        epochs=n_epoch,
        # class_weight=class_weights,
        validation_data=datagen_val.flow(test_dict, batch_size=batch_size, frames_per_batch=frames_per_batch, skip=skip),
        validation_steps=(test_dict['X'].shape[0] * test_dict['X'].shape[time_axis] // frames_per_batch) // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_sample_movie(model=None, dataset=None, optimizer=None,
                             expt='', it=0, batch_size=32, n_epoch=100,
                             window_size=(30, 30, 3),
                             balance_classes=True, max_class_samples=None,
                             direc_save='/data/models', direc_data='/data/npz_data',
                             lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                             rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='sample')
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.SampleMovieDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    datagen_val = generators.SampleMovieDataGenerator(
        rotation_range=0,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=0,  # randomly flip images
        vertical_flip=0)  # randomly flip images

    train_data = datagen.flow(
        train_dict,
        batch_size=batch_size,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples)

    val_data = datagen_val.flow(
        test_dict,
        batch_size=batch_size,
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
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    np.savez(file_name_save_loss, loss_history=loss_history.history)


def train_model_sample_watershed_movie(model=None, dataset=None, optimizer=None,
                                       expt='', it=0, batch_size=32, n_epoch=100,
                                       window_size=(30, 30, 3),
                                       balance_classes=True, max_class_samples=None,
                                       direc_save='/data/models', direc_data='/data/npz_data',
                                       lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                                       rotation_range=0, flip=True, shear=0,
                                       class_weight=None, distance_bins=16,
                                       erosion_width=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='sample')
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.SampleMovieDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    datagen_val = generators.SampleMovieDataGenerator(
        rotation_range=0,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=0,  # randomly flip images
        vertical_flip=0)  # randomly flip images

    train_data = datagen.flow(
        train_dict,
        batch_size=batch_size,
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples)

    val_data = datagen_val.flow(
        test_dict,
        batch_size=batch_size,
        distance_bins=distance_bins,
        erosion_width=erosion_width,
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
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    np.savez(file_name_save_loss, loss_history=loss_history.history)


def train_model_watershed_3D(model=None, dataset=None, optimizer=None,
                             expt='', it=0, batch_size=1, n_epoch=100,
                             direc_save='/data/models', direc_data='/data/npz_data',
                             lr_sched=rate_scheduler(lr=0.01, decay=0.95), skip=None,
                             frames_per_batch=10, rotation_range=0, flip=True,
                             shear=0, class_weight=None, distance_bins=16,
                             erosion_width=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, test_dict = get_data(training_data_file_name, mode='conv')

    if class_weight is None:
        class_weight = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        # TODO: implement direction loss
        pass

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = generators.WatershedMovieDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # no augmentation for validation data
    datagen_val = generators.WatershedMovieDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    time_axis = 2 if CHANNELS_FIRST else 1
    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, distance_bins=distance_bins, frames_per_batch=frames_per_batch, erosion_width=erosion_width, skip=skip),
        steps_per_epoch=(train_dict['y'].shape[0] * train_dict['y'].shape[time_axis] // frames_per_batch) // batch_size,
        epochs=n_epoch,
        class_weight=class_weight,
        validation_data=datagen_val.flow(test_dict, batch_size=batch_size, distance_bins=distance_bins, frames_per_batch=frames_per_batch, erosion_width=erosion_width, skip=skip),
        validation_steps=(test_dict['X'].shape[0] * test_dict['X'].shape[time_axis] // frames_per_batch) // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model
