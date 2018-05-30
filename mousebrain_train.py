## Generate training data
import os
import pdb
import platform
import argparse

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD

import numpy as np

from deepcell import make_training_data
from deepcell import bn_dense_multires_feature_net_3D as the_model
# from deepcell import bn_dense_feature_net_3D as the_model
from deepcell import rate_scheduler, train_model_movie as train_model

ON_SERVER = platform.system() == 'Linux'
DATA_DIR = '/data/training_data/nuclear' if ON_SERVER else '/Users/Will/vanvalenlab/data'
DATA_FILE = 'MouseBrain_{}'.format(K.image_data_format())

def generate_training_data():
    direc_name = os.path.join(DATA_DIR, 'MouseBrain')
    training_direcs = ['set6'] # only set6 from MouseBrain has been annotated
    raw_image_direc = 'Stack'
    annotation_direc = 'Annotation/all_montages'
    file_name_save = os.path.join(direc_name, DATA_FILE)

    # Create the training data
    make_training_data(
        dimensionality=3,
        direc_name=direc_name,
        file_name_save=file_name_save,
        channel_names=['slice'], # for iterating over stacks of images from a montage
        training_direcs=training_direcs,
        window_size_x=30,
        window_size_y=30,
        border_mode='same',
        output_mode='sample',
        reshape_size=None,
        process=True,
        process_std=True,
        display=False,
        num_frames=30,
        num_of_frames_to_display=5,
        verbose=True,
        annotation_name='', # basically channel name but for annotated images
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc)


def train_model_on_training_data():
    dataset = 'MouseBrain'
    direc_save = os.path.join(DATA_DIR, dataset)
    direc_data = os.path.join(DATA_DIR, dataset)
    number_of_frames = 10
    batch_size = 1
    n_epoch = 1

    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))
    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    if K.image_data_format() == 'channels_first':
        batch_shape = (batch_size, X.shape[1], number_of_frames, *X.shape[3:])
    else:
        batch_shape = (batch_size, number_of_frames, *X.shape[2:4], X.shape[4])

    model = the_model(batch_shape=batch_shape, permute=False)

    train_model(
        model=model,
        dataset=DATA_FILE,
        optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        batch_size=batch_size,
        number_of_frames=number_of_frames,
        n_epoch=n_epoch,
        direc_save=direc_save,
        direc_data=direc_data,
        lr_sched=rate_scheduler(lr=0.01, decay=0.95),
        rotation_range=180,
        flip=True,
        shear=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        help='force re-write of training data npz files')

    args = parser.parse_args()
    write_data = not os.path.isfile(os.path.join(DATA_DIR, 'MouseBrain', DATA_FILE + '.npz')) or args.overwrite
    if write_data:
        generate_training_data()

    train_model_on_training_data()
