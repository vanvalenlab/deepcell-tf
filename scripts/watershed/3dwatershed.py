## Generate training data
import os
import errno
import argparse

import numpy as np
from tensorflow.python.keras.optimizers import SGD,Adam
from tensorflow.python.keras import backend as K

from deepcell import get_image_sizes
from deepcell import make_training_data
from deepcell import bn_feature_net_3D as the_model
from deepcell import rate_scheduler
from deepcell import train_model_3dwatershed as train_model
from deepcell import run_models_on_directory
from deepcell import export_model

# data options
DATA_OUTPUT_MODE = 'conv'
# DATA_OUTPUT_MODE = 'sample'
BORDER_MODE = 'valid' if DATA_OUTPUT_MODE == 'sample' else 'same'
RESIZE = False
RESHAPE_SIZE = 512
NUM_FRAMES = 10

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'
PREFIX = 'cells/unspecified_nuclear_data/nuclei_broad'
DATA_FILE = 'nuclei_broad_watershed_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)

for d in (NPZ_DIR, MODEL_DIR, RESULTS_DIR):
    try:
        os.makedirs(os.path.join(d, PREFIX))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

def generate_training_data():
    direc_name = os.path.join(DATA_DIR, PREFIX)
    training_direcs = ['set6'] # only set6 from MouseBrain has been annotated
    raw_image_direc = 'stacked_raw'
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
        border_mode=BORDER_MODE,
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
    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))

    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    n_epoch = 16
    batch_size = 1
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    frames_per_batch = 10

    distance_bins = 16

    model_args = {
        'norm_method': 'whole_image',
        'n_features': distance_bins
    }

    data_format = K.image_data_format()
    row_axis = 3 if data_format == 'channels_first' else 2
    col_axis = 4 if data_format == 'channels_first' else 3
    channel_axis = 1 if data_format == 'channels_first' else 4

    nrow, ncol = X.shape[row_axis:col_axis + 1] if not RESIZE else (RESHAPE_SIZE, RESHAPE_SIZE)
    if data_format == 'channels_first':
        batch_shape = (batch_size, X.shape[channel_axis], frames_per_batch, nrow, ncol)
    else:
        batch_shape = (batch_size, frames_per_batch, nrow, ncol, X.shape[channel_axis])
    model_args['batch_shape'] = batch_shape

    model = the_model(**model_args)

    train_model(
        model=model,
        dataset=DATA_FILE,
        optimizer=sgd,
        batch_size=batch_size,
        number_of_frames=frames_per_batch,
        n_epoch=n_epoch,
        direc_save=direc_save,
        direc_data=direc_data,
        lr_sched=rate_scheduler(lr=0.01, decay=0.95),
        rotation_range=180,
        flip=True,
        shear=0)


def run_model_on_dir():
    raw_dir = 'raw'
    data_location = os.path.join(DATA_DIR, PREFIX, 'set1', raw_dir)
    output_location = os.path.join(RESULTS_DIR, PREFIX)
    channel_names = ['nuclear']
    image_size_x, image_size_y = get_image_sizes(data_location, channel_names)

    model_name = '2018-07-01_nuclei_broad_watershed_channels_last_conv__0.h5'

    weights = os.path.join(MODEL_DIR, PREFIX, model_name)

    n_features = 4
    window_size = (30, 30)

    if DATA_OUTPUT_MODE == 'sample':
        model_fn = dilated_bn_feature_net_61x61
    elif DATA_OUTPUT_MODE == 'conv':
        model_fn = bn_dense_feature_net

    predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=channel_names,
        output_location=output_location,
        n_features=n_features,
        model_fn=model_fn,
        list_of_weights=[weights],
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        win_x=window_size[0],
        win_y=window_size[1],
        split=False)

def export():
    model_args = {
        'norm_method': 'median',
        'reg': 1e-5,
        'n_features': 3
    }

    direc_data = os.path.join(NPZ_DIR, PREFIX)
    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))
    X, y = training_data['X'], training_data['y']

    data_format = K.image_data_format()
    row_axis = 2 if data_format == 'channels_first' else 1
    col_axis = 3 if data_format == 'channels_first' else 2
    channel_axis = 1 if data_format == 'channels_first' else 3

    if DATA_OUTPUT_MODE == 'sample':
        the_model = watershednetwork
        if K.image_data_format() == 'channels_first':
            model_args['input_shape'] = (1, 1080, 1280)
        else:
            model_args['input_shape'] = (1080, 1280, 1)

    elif DATA_OUTPUT_MODE == 'conv' or DATA_OUTPUT_MODE == 'disc':
        the_model = watershednetwork
        model_args['location'] = False

        size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
        if data_format == 'channels_first':
            model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
        else:
            model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])

    model = the_model(**model_args)

    model_name = '2018-06-29_ecoli_watershed_channels_last_conv__0.h5'

    weights_path = os.path.join(MODEL_DIR, PREFIX, model_name)
    export_path = os.path.join(EXPORT_DIR, PREFIX)
    export_model(model, export_path, model_version=0, weights_path=weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=['train', 'run', 'export'],
                        help='train or run models')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        help='force re-write of training data npz files')

    args = parser.parse_args()

    if args.command == 'train':
        data_file_exists = os.path.isfile(os.path.join(NPZ_DIR, PREFIX, DATA_FILE + '.npz'))
        if args.overwrite or not data_file_exists:
            generate_training_data()

        train_model_on_training_data()

    elif args.command == 'run':
        run_model_on_dir()

    elif args.command == 'export':
        export()
