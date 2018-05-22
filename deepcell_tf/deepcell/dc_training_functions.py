"""
dc_training_functions.py

Functions for training convolutional neural networks

@author: David Van Valen
"""

from __future__ import print_function
from __future__ import division

import datetime
import os

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import tifffile as tiff
from .dc_helper_functions import *
from .dc_image_generators import *

"""
Training convnets
"""

def train_model_sample(model=None, dataset=None, optimizer=None,
    expt="", it=0, batch_size=32, n_epoch=100,
    direc_save="/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/",
    direc_data="/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/",
    lr_sched=rate_scheduler(lr = 0.01, decay = 0.95),
    rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + ".npz")
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))

    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, Y_test) = get_data(training_data_file_name)

    # move the channels to the last dimension if needed
    if K.image_data_format() == 'channels_last':
        X_test = np.rollaxis(X_test, 1, 4)

    # the data, shuffled and split between train and test sets
    print(('X_train shape:', train_dict['X'].shape))
    print((train_dict["pixels_x"].shape[0], 'train samples'))
    print((X_test.shape[0], 'test samples'))

    # determine the number of classes
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[1]

    print(output_shape, n_classes)

    # convert class vectors to binary class matrices
    train_dict['y'] = to_categorical(train_dict['y'], n_classes)
    Y_test = to_categorical(Y_test, n_classes)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = SampleDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(datagen.sample_flow(train_dict, batch_size = batch_size),
                        steps_per_epoch=len(train_dict['y']) // batch_size,
                        epochs=n_epoch,
                        validation_data=(X_test, Y_test),
                        validation_steps=X_test.shape[0] // batch_size,
                        class_weight=class_weight,
                        callbacks=[ModelCheckpoint(file_name_save, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                                   LearningRateScheduler(lr_sched)])

    np.savez(file_name_save_loss, loss_history = loss_history.history)

def train_model_conv(model=None, dataset=None, optimizer=None,
    expt="", it=0, batch_size=1, n_epoch=100,
    direc_save="/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/",
    direc_data="/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/",
    lr_sched=rate_scheduler(lr = 0.01, decay = 0.95),
    rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + ".npz")
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

    file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

    train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode='conv')

    class_weights = None #class_weight #train_dict["class_weights"]

    # the data, shuffled and split between train and test sets
    print(('Training data shape:', train_dict['X'].shape))
    print(('Training labels shape:', train_dict['y'].shape))

    print(('Testing data shape:', X_test.shape))
    print(('Testing labels shape:', Y_test.shape))

    # determine the number of classes
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    print(output_shape, n_classes)
    print(class_weights)

    def loss_function(y_true, y_pred):
        return weighted_categorical_crossentropy(y_true, y_pred, n_classes=n_classes, from_logits=False)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageFullyConvDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # move the channels to the last dimension if needed
    if K.image_data_format() == 'channels_last':
        X_test = np.rollaxis(X_test, 1, 4)

    Y_test = np.rollaxis(Y_test, 1, 4)


    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(datagen.flow(train_dict, batch_size=batch_size),
                        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
                        epochs=n_epoch,
                        validation_data=(X_test, Y_test),
                        validation_steps=X_test.shape[0] // batch_size,
                        callbacks=[ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto'),
                                   LearningRateScheduler(lr_sched)])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model

def train_model_watershed(model=None, dataset=None, optimizer=None,
    expt="", it=0, batch_size=1, n_epoch=100,
    direc_save="/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/",
    direc_data="/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/",
    lr_sched=rate_scheduler(lr=0.01, decay=0.95),
    rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + ".npz")
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

    file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

    train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode = 'conv')

    class_weights = None #class_weight #train_dict["class_weights"]

    # the data, shuffled and split between train and test sets
    print(('Training data shape:', train_dict['X'].shape))
    print(('Training labels shape:', train_dict['y'].shape))

    print(('Testing data shape:', X_test.shape))
    print(('Testing labels shape:', Y_test.shape))

    # determine the number of classes
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    print(output_shape, n_classes)
    print(class_weights)

    def loss_function(y_true, y_pred):
        return direction_loss(y_true, y_pred)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageFullyConvDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    # move the channels to the last dimension if needed
    if K.image_data_format() == 'channels_last':
        X_test = np.rollaxis(X_test, 1, 4)

    Y_test = np.rollaxis(Y_test, 1, 4)


    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(datagen.flow(train_dict, batch_size=batch_size, target_format='direction'),
                        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
                        epochs=n_epoch,
                        validation_data=(X_test, Y_test),
                        validation_steps=X_test.shape[0] // batch_size,
                        callbacks=[ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                                   LearningRateScheduler(lr_sched)])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model

def train_model_disc(model=None, dataset=None, optimizer=None,
    expt="", it=0, batch_size=1, n_epoch=100,
    direc_save="/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/",
    direc_data="/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/",
    lr_sched=rate_scheduler(lr=0.01, decay=0.95),
    rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + ".npz")
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

    file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

    train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode='conv')

    # the data, shuffled and split between train and test sets
    print('Training data shape:', train_dict['X'].shape)
    print('Training labels shape:', train_dict['y'].shape)

    print('Testing data shape:', X_test.shape)
    print('Testing labels shape:', Y_test.shape)

    # determine the number of classes
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    print(output_shape, n_classes)

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

    Y_test = np.rollaxis(Y_test, 1, 4)

    loss_history = model.fit_generator(datagen.flow(train_dict, batch_size=batch_size),
                        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
                        epochs=n_epoch,
                        validation_data=(X_test, Y_test),
                        validation_steps=X_test.shape[0] // batch_size,
                        callbacks=[ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto'),
                                   LearningRateScheduler(lr_sched)])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model


def train_model_conv_sample(model=None, dataset=None, optimizer=None,
    expt="", it=0, batch_size=1, n_epoch=100,
    direc_save="/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/",
    direc_data="/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/",
    lr_sched=rate_scheduler(lr=0.01, decay=0.95),
    rotation_range=0, flip=True, shear=0, class_weights=None):

    training_data_file_name = os.path.join(direc_data, dataset + ".npz")
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

    file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

    train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode='conv_sample')

    class_weights = class_weights #train_dict["class_weights"]
    # the data, shuffled and split between train and test sets
    print('Training data shape:', train_dict['X'].shape)
    print('Training labels shape:', train_dict['y'].shape)

    print('Testing data shape:', X_test.shape)
    print('Testing labels shape:', Y_test.shape)

    # determine the number of classes
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    print(output_shape, n_classes)

    class_weights = np.array([1, 1, 1], dtype = K.floatx())
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

    Y_test = np.rollaxis(Y_test, 1, 4)


    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(datagen.flow(train_dict, batch_size=batch_size),
                        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
                        epochs=n_epoch,
                        validation_data=(X_test, Y_test),
                        validation_steps=X_test.shape[0] // batch_size,
                        callbacks=[ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                                   LearningRateScheduler(lr_sched)])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    data_location = '/home/vanvalen/Data/RAW_40X_tube/set1/'
    channel_names = ["channel004", "channel001"]
    image_list = get_images_from_directory(data_location, channel_names)
    image = image_list[0]
    for j in range(image.shape[1]):
            image[0, j, :, :] = process_image(image[0, j, :, :], 30, 30, False)

    pred = model.predict(image)
    for j in range(3):
        save_name = 'feature_' + str(j) + '.tiff'
        tiff.imsave(save_name, pred[0, :, :, j])

    return model

def train_model_movie(model=None, dataset=None, optimizer=None,
    expt="", it=0, batch_size=1, n_epoch=100,
    direc_save="/data/trained_networks/nuclear_movie",
    direc_data="/data/training_data_npz/nuclear_movie",
    lr_sched=rate_scheduler(lr=0.01, decay=0.95),
    number_of_frames=10,
    rotation_range=0, flip=True, shear=0, class_weight=None):

    training_data_file_name = os.path.join(direc_data, dataset + ".npz")
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

    file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

    train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode = 'movie')

    class_weights = None #class_weight #train_dict["class_weights"]
    # the data, shuffled and split between train and test sets
    print('Training data shape:', train_dict['X'].shape)
    print('Training labels shape:', train_dict['y'].shape)

    print('Testing data shape:', X_test.shape)
    print('Testing labels shape:', Y_test.shape)

    # determine the number of classes
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    print(output_shape, n_classes)

    def loss_function(y_true, y_pred):
        return discriminative_instance_loss_3D(y_true, y_pred)

    model.compile(loss=loss_function, optimizer=optimizer)

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = MovieDataGenerator(
        rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=flip,  # randomly flip images
        vertical_flip=flip)  # randomly flip images

    print(train_dict['X'].shape)

    X_test = X_test[:, :, 0:number_of_frames, :, :]
    Y_test = Y_test[:, :, 0:number_of_frames, :, :]
    Y_test = np.rollaxis(Y_test, 1, 5)


    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(datagen.flow(train_dict, batch_size=batch_size, number_of_frames=number_of_frames),
                        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
                        epochs=n_epoch,
                        validation_data=(X_test, Y_test),
                        validation_steps=X_test.shape[0] // batch_size,
                        callbacks=[ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                                   LearningRateScheduler(lr_sched)])

    model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    return model
