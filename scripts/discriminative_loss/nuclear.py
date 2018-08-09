# Import packages
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import warnings
import errno
import numpy as np
import datetime
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.filters import sobel_h
from skimage.filters import sobel_v
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.external import tifffile as tiff
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import random_channel_shift
from tensorflow.python.keras.preprocessing.image import apply_transform
from tensorflow.python.keras.preprocessing.image import flip_axis
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import to_categorical as keras_to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.optimizers import SGD,Adam

from deepcell import get_image_sizes
from deepcell import make_training_data
from deepcell import rate_scheduler
from deepcell import bn_multires_feature_net
from deepcell import disc_net
from deepcell import discriminative_instance_loss
from deepcell import DiscDataGenerator
from deepcell.utils.transform_utils import transform_matrix_offset_center
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.image_generators import ImageFullyConvDataGenerator
from deepcell.losses import weighted_categorical_crossentropy
from deepcell.utils.io_utils import get_images_from_directory
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler
from deepcell.utils.transform_utils import to_categorical
from deepcell.settings import CHANNELS_FIRST


# data options
DATA_OUTPUT_MODE = 'conv'
PADDING = 'valid' if DATA_OUTPUT_MODE == 'sample' else 'same'
RESIZE = True
RESHAPE_SIZE = 256

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'
PREFIX = 'cells/unspecified_nuclear_data/nuclei/'
DATA_FILE = 'disc_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)

for d in (NPZ_DIR, MODEL_DIR, RESULTS_DIR):
    try:
        os.makedirs(os.path.join(d, PREFIX))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
            
def generate_training_data():
    file_name_save = os.path.join(NPZ_DIR, PREFIX, DATA_FILE)
    num_of_features = 2 # Specify the number of feature masks that are present
    window_size = (30, 30) # Size of window around pixel
    training_direcs = ['set1', 'set2', 'set3', 'set4', 'set5', 'set6', 'set7', 'set8', 'set9']
    channel_names = ['nuclear']
    raw_image_direc = ''
    annotation_direc = ''

    # Create the training data
    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX),
        dimensionality=2,
        max_training_examples=1e6, # Define maximum number of training examples
        window_size_x=window_size[0],
        window_size_y=window_size[1],
        padding=PADDING,
        file_name_save=file_name_save,
        training_direcs=training_direcs,
        channel_names=channel_names,
        num_of_features=num_of_features,
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc,
        reshape_size=RESHAPE_SIZE if RESIZE else None,
        edge_feature=[1, 0, 0], # Specify which feature is the edge feature,
        dilation_radius=1,
        output_mode=DATA_OUTPUT_MODE,
        display=False,
        verbose=True)

    
def train_model_on_training_data():
    # Foreground and background training with cross entropy loss
    direc_save = os.path.join(MODEL_DIR, PREFIX)
    direc_data = os.path.join(NPZ_DIR, PREFIX)
    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))

    class_weights = training_data['class_weights']
    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    n_epoch = 10
    batch_size = 32 if DATA_OUTPUT_MODE == 'sample' else 1
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    lr_sched = rate_scheduler(lr=0.01, decay=0.99)

    model_args = {
    }

    data_format = K.image_data_format()
    row_axis = 2 if data_format == 'channels_first' else 1
    col_axis = 3 if data_format == 'channels_first' else 2
    channel_axis = 1 if data_format == 'channels_first' else 3

    the_model = bn_multires_feature_net

    size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
    if data_format == 'channels_first':
        model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
    else:
        model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])
    model_args['n_features'] = 3
    model_args['location'] = False

    fgbg_model = the_model(**model_args)

    dataset = DATA_FILE
    expt = 'fgbg'
    it = 0
    training_data_file_name = os.path.join(direc_data, dataset + '.npz')

    file_name_save = os.path.join(direc_save, '{}_fgbg_{}_{}.h5'.format(todays_date, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_fgbg_{}_{}.npz'.format(todays_date, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='conv')

    test_dict = {
        'X':X_test,
        'y':y_test,
        'class_weights':train_dict['class_weights'],
        'win_x':train_dict['win_x'],
        'win_y':train_dict['win_y']
    }

    class_weights = train_dict['class_weights']
    n_classes = fgbg_model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', fgbg_model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function_fgbg(y_true, y_pred):
        return weighted_categorical_crossentropy(y_true, y_pred,
                                                 n_classes=n_classes,
                                                 from_logits=False)

    fgbg_model.compile(loss=loss_function_fgbg, optimizer=optimizer, metrics=['accuracy'])

    datagen_fgbg = ImageFullyConvDataGenerator(
        rotation_range=180,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    datagen_fgbg_test = ImageFullyConvDataGenerator(
        rotation_range=0,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    loss_history = fgbg_model.fit_generator(
        datagen_fgbg.flow(train_dict, batch_size=batch_size),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=datagen_fgbg_test.flow(test_dict, batch_size=batch_size),
        validation_steps=X_test.shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])
    
    fgbg_model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)


    # Set up disc model
    direc_save = os.path.join(MODEL_DIR, PREFIX)
    direc_data = os.path.join(NPZ_DIR, PREFIX)
    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))

    class_weights = training_data['class_weights']
    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    n_epoch = 10
    batch_size = 32 if DATA_OUTPUT_MODE == 'sample' else 1
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    lr_sched = rate_scheduler(lr=0.01, decay=0.99)

    data_format = K.image_data_format()
    row_axis = 2 if data_format == 'channels_first' else 1
    col_axis = 3 if data_format == 'channels_first' else 2
    channel_axis = 1 if data_format == 'channels_first' else 3

    model_args_disc_net = {}
    size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
    if data_format == 'channels_first':
        model_args_disc_net['input_shape'] = (X.shape[channel_axis], size[0], size[1])
    else:
        model_args_disc_net['input_shape'] = (size[0], size[1], X.shape[channel_axis])
    model_args_disc_net['norm_method'] = 'std'
    model_args_disc_net['n_features'] = 2
    model_args_disc_net['seg_model'] = fgbg_model
    model_args_disc_net['softmax'] = False

    disc_model = disc_net(**model_args_disc_net)


    # Train with disc loss
    dataset = DATA_FILE
    expt = ''
    it = 0
    n_epoch = 10
    lr_sched = rate_scheduler(lr = 0.01, decay=0.95)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    training_data_file_name = os.path.join(direc_data, dataset + '.npz')

    file_name_save = os.path.join(direc_save, '{}_{}_{}_{}.h5'.format(todays_date, dataset, expt, it))
    file_name_save_loss = os.path.join(direc_save, '{}_{}_{}_{}.npz'.format(todays_date, dataset, expt, it))

    train_dict, (X_test, y_test) = get_data(training_data_file_name, mode='conv')

    test_dict = {
        'X':X_test,
        'y':y_test,
        'class_weights':train_dict['class_weights'],
        'win_x':train_dict['win_x'],
        'win_y':train_dict['win_y']
    }

    class_weights = train_dict['class_weights']
    n_classes = disc_model.layers[-1].output_shape[1 if CHANNELS_FIRST else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('Output Shape:', disc_model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        return discriminative_instance_loss(y_true, y_pred)

    disc_model.compile(loss=loss_function, optimizer=optimizer)


    datagen = DiscDataGenerator(
        rotation_range=180,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    datagen_test = DiscDataGenerator(
        rotation_range=0,  # randomly rotate images by 0 to rotation_range degrees
        shear_range=0, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    loss_history = disc_model.fit_generator(
        datagen.flow(train_dict, batch_size=batch_size),
        steps_per_epoch=train_dict['y'].shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=datagen_test.flow(test_dict, batch_size=batch_size),
        validation_steps=X_test.shape[0] // batch_size,
        callbacks=[
            ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            LearningRateScheduler(lr_sched)
        ])
    
    disc_model.save_weights(file_name_save)
    np.savez(file_name_save_loss, loss_history=loss_history.history)

    
    # Save images
    test_images = disc_model.predict(X_test)
    test_images_fgbg = fgbg_model.predict(X_test)
    print('Test_images shape:', test_images.shape)

    test_images_post_fgbg = test_images[:,:,:,:] * np.expand_dims(test_images_fgbg[:,:,:,1]>0.8,axis=-1)

    output_location = os.path.join(RESULTS_DIR, PREFIX)

    for index in range(test_images.shape[0]):
        for channel in range(test_images.shape[-1]):
            cnnout_name_fgbg = 'test_images_post_fgbg_{}_{}.tif'.format(index, channel)
            tiff.imsave(os.path.join(output_location, cnnout_name_fgbg), test_images_fgbg[index,:,:,channel])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        help='force re-write of training data npz files')

    args = parser.parse_args()
    
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d')

    data_file_exists = os.path.isfile(os.path.join(NPZ_DIR, PREFIX, DATA_FILE + '.npz'))
    if args.overwrite or not data_file_exists:
        generate_training_data()

    train_model_on_training_data()
