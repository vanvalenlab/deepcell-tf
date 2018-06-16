## Generate training data
import os
import errno
import argparse

import numpy as np
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import backend as K

from deepcell import get_image_sizes
from deepcell import make_training_data
from deepcell import bn_feature_net_61x61
from deepcell import bn_dense_feature_net
from deepcell import rate_scheduler
from deepcell import train_model_disc, train_model_conv, train_model_sample
from deepcell import run_models_on_directory
from deepcell import export_model

# data options
DATA_OUTPUT_MODE = 'conv'
DATA_OUTPUT_MODE = 'sample'
BORDER_MODE = 'valid' if DATA_OUTPUT_MODE == 'sample' else 'same'
RESIZE = True
RESHAPE_SIZE = 512

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'
PREFIX = 'cells/ecoli/kc_polaris'
DATA_FILE = 'ecoli_kc_polaris_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)

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
    training_direcs = ['set1', 'set2']
    channel_names = ['Phase']
    raw_image_direc = 'raw'
    annotation_direc = 'annotated'

    # Create the training data
    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX),
        dimensionality=2,
        max_training_examples=1e6, # Define maximum number of training examples
        window_size_x=window_size[0],
        window_size_y=window_size[1],
        border_mode=BORDER_MODE,
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
    direc_save = os.path.join(MODEL_DIR, PREFIX)
    direc_data = os.path.join(NPZ_DIR, PREFIX)
    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))

    class_weights = training_data['class_weights']
    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    n_epoch = 100
    batch_size = 32 if DATA_OUTPUT_MODE == 'sample' else 1
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    lr_sched = rate_scheduler(lr=0.01, decay=0.99)

    model_args = {
        'norm_method': 'median',
        'reg': 1e-5,
        'n_features': 3
    }

    data_format = K.image_data_format()
    row_axis = 2 if data_format == 'channels_first' else 1
    col_axis = 3 if data_format == 'channels_first' else 2
    channel_axis = 1 if data_format == 'channels_first' else 3

    if DATA_OUTPUT_MODE == 'sample':
        train_model = train_model_sample
        the_model = bn_feature_net_61x61
        model_args['n_channels'] = 1

    elif DATA_OUTPUT_MODE == 'conv' or DATA_OUTPUT_MODE == 'disc':
        train_model = train_model_conv
        the_model = bn_dense_feature_net
        model_args['location'] = False

        size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
        if data_format == 'channels_first':
            model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
        else:
            model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])

    model = the_model(**model_args)

    train_model(
        model=model,
        dataset=DATA_FILE,
        optimizer=optimizer,
        batch_size=batch_size,
        n_epoch=n_epoch,
        direc_save=direc_save,
        direc_data=direc_data,
        lr_sched=lr_sched,
        class_weight=class_weights,
        rotation_range=180,
        flip=True,
        shear=True)


def run_model_on_dir():
    data_location = os.path.join(DATA_DIR, PREFIX, 'set1')
    output_location = os.path.join(RESULTS_DIR, PREFIX)
    channel_names = ['phase']
    image_size_x, image_size_y = get_image_sizes(data_location, channel_names)

    model_name = '2018-06-13_ecoli_kc_polaris_channels_last_sample__0.h5'
    weights = os.path.join(MODEL_DIR, PREFIX, model_name)

    n_features = 3
    window_size = (30, 30)

    model_fn = bn_feature_net_61x61 if DATA_OUTPUT_MODE == 'sample' else bn_dense_feature_net

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
        the_model = bn_feature_net_61x61
        model_args['n_channels'] = 1

    elif DATA_OUTPUT_MODE == 'conv' or DATA_OUTPUT_MODE == 'disc':
        the_model = bn_dense_feature_net
        model_args['location'] = False

        size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
        if data_format == 'channels_first':
            model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
        else:
            model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])

    the_model = bn_feature_net_61x61 if DATA_OUTPUT_MODE == 'sample' else bn_dense_feature_net
    model = the_model(**model_args)

    model_name = '2018-06-13_ecoli_kc_polaris_channels_last_sample__0.h5'
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
