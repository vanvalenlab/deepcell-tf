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
from skimage.external import tifffile as tiff
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical as keras_to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from .image_generators import SampleDataGenerator
from .image_generators import ImageFullyConvDataGenerator
from .image_generators import MovieDataGenerator
from .image_generators import SiameseDataGenerator
from .image_generators import WatershedDataGenerator
from .image_generators import WatershedSampleDataGenerator
from .losses import sample_categorical_crossentropy
from .losses import weighted_categorical_crossentropy
from .losses import discriminative_instance_loss
from .losses import discriminative_instance_loss_3D
from .utils.io_utils import get_images_from_directory
from .utils.data_utils import get_data
from .utils.train_utils import rate_scheduler
from .utils.transform_utils import to_categorical
from .settings import CHANNELS_FIRST

"""
Training convnets
"""

def train_model_sample(model=None, dataset=None, optimizer=None,
                       expt='', it=0, batch_size=32, n_epoch=100,
                       direc_save='/data/models', direc_data='/data/npz_data',
                       lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                       rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='sample')
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('pixels_x shape:', train_dict['pixels_x'].shape[0])
    print('X_test shape:', X_test.shape[0])
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    # convert class vectors to binary class matrices
    train_dict['y'] = to_categorical(train_dict['y'], n_classes)
    y_test = to_categorical(y_test, n_classes)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = SampleDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.sample_flow(train_dict, batch_size=batch_size),
        steps_per_epoch=len(train_dict['y']) // batch_size,
        epochs=n_epoch,
        validation_data=(X_test, y_test),
        validation_steps=X_test.shape[0] // batch_size,
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    np.savez(file_name_save_loss, loss_history=loss_history.history)

def train_model_conv(model=None, dataset=None, optimizer=None,
                     expt='', it=0, batch_size=1, n_epoch=100,
                     direc_save='/data/models', direc_data='/data/npz_data',
                     lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                     rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='conv')

    class_weights = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return weighted_categorical_crossentropy(y_true, y_pred,
                                                 n_classes=n_classes,
                                                 from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageFullyConvDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=(X_test, y_test),
        validation_steps=X_test.shape[0] // batch_size,
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

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='siamese')

    class_weights = train_dict['class_weights']
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', model.layers[-1].output_shape)

    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]

    def loss_function(y_true, y_pred):
        return weighted_categorical_crossentropy(y_true, y_pred,
                                                 n_classes=n_classes,
                                                 from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = SiameseDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    datagen_val = SiameseDataGenerator(
        rotation_range=0,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=0,  # randomly flip images
        vertical_flip=0)  # randomly flip images

    validation_dict = {'X': X_test, 'y': y_test}

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.siamese_flow(train_dict, batch_size=batch_size),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=datagen_val.siamese_flow(validation_dict, batch_size=batch_size),
        validation_steps=X_test.shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model

def train_model_watershed(model=None, dataset=None, optimizer=None,
                          expt='', it=0, batch_size=1, n_epoch=100, distance_bins=16,
                          direc_save='/data/models', direc_data='/data/npz_data',
                          lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                          rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='conv')

    if class_weight is None:
        class_weight = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        # TODO: implement direction loss
        pass

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = WatershedDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # no augmentation for training data
    datagen_val = WatershedDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    validation_dict = {'X': X_test, 'y': y_test}

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, distance_bins=distance_bins),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        class_weight=class_weight,
        validation_data=datagen_val.flow(validation_dict, batch_size=batch_size, distance_bins=distance_bins),
        validation_steps=X_test.shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model

def train_model_watershed_sample(model=None, dataset=None, optimizer=None,
                                 expt='', it=0, batch_size=1, n_epoch=100, distance_bins=16,
                                 direc_save='/data/models', direc_data='/data/npz_data',
                                 lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                                 rotation_range=0, flip=True, shear=0, class_weight=None):
    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='conv')

    if class_weight is None:
        class_weight = train_dict['class_weights']

    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        # TODO: implement direction loss
        pass

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = WatershedSampleDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear,  # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # no augmentation for training data
    datagen_val = WatershedSampleDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    validation_dict = {'X': X_test, 'y': y_test}

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, distance_bins=distance_bins),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        class_weight=class_weight,
        validation_data=datagen_val.flow(validation_dict, batch_size=batch_size, distance_bins=distance_bins),
        validation_steps=X_test.shape[0] // batch_size,
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
                     rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='conv')
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return discriminative_instance_loss(y_true, y_pred)

    model.compile(loss=loss_function, optimizer=optimizer)

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageFullyConvDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=(X_test, y_test),
        validation_steps=X_test.shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model

def train_model_conv_sample(model=None, dataset=None, optimizer=None,
                            expt='', it=0, batch_size=1, n_epoch=100,
                            direc_save='/data/models', direc_data='/data/npz_data',
                            lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                            rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + '.npz')
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='conv_sample')

    class_weights = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    class_weights = np.array([1, 1, 1], dtype=K.floatx())
    def loss_function(y_true, y_pred):
        return sample_categorical_crossentropy(y_true, y_pred, axis=3,
                                               class_weights=class_weights, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageFullyConvDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    x, y = next(datagen.flow(train_dict, batch_size=1))

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=(X_test, y_test),
        validation_steps=X_test.shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    data_location = '/home/vanvalen/Data/RAW_40X_tube/set1/'
    channel_names = ['channel004', 'channel001']
    image_list = get_images_from_directory(data_location, channel_names)
    image = image_list[0]
    # for j in range(image.shape[1]):
    #     image[0, j, :, :] = process_image(image[0, j, :, :], 30, 30, False)

    pred = model.predict(image)
    for j in range(3):
        save_name = 'feature_{}.tiff'.format(j)
        tiff.imsave(save_name, pred[0, :, :, j])

    return model

def train_model_movie(model=None, dataset=None, optimizer=None,
                      expt='', it=0, batch_size=1, n_epoch=100,
                      direc_save='/data/models', direc_data='/data/npz_data',
                      lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                      number_of_frames=10, rotation_range=0, flip=True,
                      shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, '{}.npz'.format(dataset))
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='movie')
    X_train, y_train = train_dict['X'], train_dict['y']

    class_weights = train_dict['class_weights']
    n_classes = model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        # return discriminative_instance_loss_3D(y_true, y_pred)
        return weighted_categorical_crossentropy(y_true, y_pred,
                                                 n_classes=n_classes,
                                                 from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = MovieDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # no augmentation for validation data
    datagen_val = MovieDataGenerator(
        rotation_range=0,
        shear_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    # set all cell IDs to 1.  We care about is/is not a cell, not the ID
    train_dict['y'][train_dict['y'] > 0] = 1
    y_test[y_test > 0] = 1

    if class_weights is None:
        class_weights = compute_class_weight(
            'balanced',
            y=train_dict['y'].reshape(train_dict['y'].size),
            classes=np.unique(train_dict['y']))

    # keras to_categorical will not work with channels_first data.
    # if channels_first, convert to channels_last
    if K.image_data_format() == 'channels_first':
        train_dict['y'] = np.rollaxis(train_dict['y'], 1, 5)
        y_test = np.rollaxis(y_test, 1, 5)

    # use to_categorical to one-hot encode each feaeture
    train_dict['y'] = keras_to_categorical(train_dict['y'])
    y_test = keras_to_categorical(y_test)

    # if channels_first, roll axis back to expected shape
    if K.image_data_format() == 'channels_first':
        train_dict['y'] = np.rollaxis(train_dict['y'], 4, 1)
        y_test = np.rollaxis(y_test, 4, 1)

    validation_dict = {'X': X_test, 'y': y_test}

    time_axis = 2 if CHANNELS_FIRST else 1
    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size, number_of_frames=number_of_frames),
        steps_per_epoch=(train_dict['y'].shape[0] * train_dict['y'].shape[time_axis] // number_of_frames) // batch_size,
        epochs=n_epoch,
        # class_weight=class_weights,
        validation_data=datagen_val.flow(validation_dict, batch_size=batch_size, number_of_frames=number_of_frames),
        validation_steps=(X_test.shape[0] * X_test.shape[time_axis] // number_of_frames) // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model
