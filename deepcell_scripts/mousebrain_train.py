## Generate training data
import os
import pdb
import platform
import argparse

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD, Adam

import numpy as np

from deepcell import make_training_data
#from deepcell import bn_dense_multires_feature_net_3D as the_model
#from deepcell import bn_dense_feature_net_lstm as the_model
from deepcell import bn_dense_feature_net_3D as the_model
#from deepcell import siamese_model as the_model
from deepcell import rate_scheduler, train_model_movie as train_model
from deepcell import nikon_getfiles
from deepcell import get_image
from deepcell import run_models_on_directory
from deepcell import get_image_sizes

# data options
DATA_OUTPUT_MODE = 'conv'
RESIZE = False
RESHAPE_SIZE = 128
NUM_FRAMES = 30

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
PREFIX = 'cells/MouseBrain/generic'
DATA_FILE = 'MouseBrain_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)


def generate_training_data():
    direc_name = os.path.join(DATA_DIR, PREFIX)
    training_direcs = ['set6'] # only set6 from MouseBrain has been annotated
    raw_image_direc = 'stacked'
    annotation_direc = 'annotated/all_montages'
    file_name_save = os.path.join(NPZ_DIR, PREFIX, DATA_FILE)

    # Create the training data
    make_training_data(
        dimensionality=3,
        direc_name=direc_name,
        file_name_save=file_name_save,
        channel_names=['slice'], # for iterating over stacks of images from a montage
        training_direcs=training_direcs,
        output_mode=DATA_OUTPUT_MODE,
        window_size_x=30,
        window_size_y=30,
        border_mode='same',
        reshape_size=None if not RESIZE else RESHAPE_SIZE,
        process=True,
        process_std=True,
        display=False,
        num_frames=NUM_FRAMES,
        num_of_frames_to_display=5,
        verbose=True,
        montage_mode=True,
        annotation_name='', # basically channel name but for annotated images
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc)


def train_model_on_training_data():
    direc_save = os.path.join(MODEL_DIR, PREFIX)
    direc_data = os.path.join(NPZ_DIR, PREFIX)
    batch_size = 1
    n_epoch = 50
    number_of_frames = 10

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))
    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    if K.image_data_format() == 'channels_first':
        row_size, col_size = X.shape[3:] if not RESIZE else (RESHAPE_SIZE, RESHAPE_SIZE)
        batch_shape = (batch_size, X.shape[1], number_of_frames, row_size, col_size)
    else:
        row_size, col_size = X.shape[2:4] if not RESIZE else (RESHAPE_SIZE, RESHAPE_SIZE)
        batch_shape = (batch_size, number_of_frames, row_size, col_size, X.shape[4])

    n_features = 2 # np.unique(y).size
    model = the_model(batch_shape=batch_shape, n_features=n_features,
                      permute=False, location=False)

    train_model(
        model=model,
        dataset=DATA_FILE,
        optimizer=sgd,
        batch_size=batch_size,
        number_of_frames=number_of_frames,
        n_epoch=n_epoch,
        direc_save=direc_save,
        direc_data=direc_data,
        lr_sched=rate_scheduler(lr=0.01, decay=0.95),
        rotation_range=0,
        flip=False,
        shear=0
    )


def run_model_on_dir():
    channel_names = ['slice']
    data_location = os.path.join(DATA_DIR, PREFIX, 'set0', 'set_0_x_0_y_0')

    image_size_x, image_size_y = get_image_sizes(data_location, channel_names)

    # Define the model
    model_name = '2018-06-12_MouseBrain_channels_last_conv__0.h5'
    weights = os.path.join(MODEL_DIR, PREFIX, model_name)

    predictions = run_models_on_directory(
        data_location, channel_names, os.path.join(RESULTS_DIR, PREFIX),
        n_features=2,
        model_fn=the_model,
        list_of_weights=[weights],
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        win_x=30,
        win_y=30,
        std=True,
        split=False)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=['train', 'run'], nargs='?',
                        default='train', help='train or run models')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        help='force re-write of training data npz files')

    args = parser.parse_args()
    data_file_exists = os.path.isfile(os.path.join(NPZ_DIR, PREFIX, DATA_FILE + '.npz'))

    if args.command == 'train':
        if args.overwrite or not data_file_exists:
            generate_training_data()

        train_model_on_training_data()

    elif args.command == 'run':
        run_model_on_dir()
