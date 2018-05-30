## Generate training data
import os
import pdb
import platform

from tensorflow.python.keras.optimizers import SGD

import numpy as np

from deepcell import make_training_data
from deepcell import bn_feature_net_61x61 as the_model
from deepcell import rate_scheduler, train_model_sample as train_model

ON_SERVER = platform.system() == 'Linux'
DATA_DIR = '/data/old_training_data' if ON_SERVER else '/Users/Will/vanvalenlab/data'

def generate_training_data():
    # Load data
    direc_name = os.path.join(DATA_DIR, 'ecoli')
    file_name_save = os.path.join(direc_name, 'ecoli_61x61.npz')

    # Create the training data
    make_training_data(
        dimensionality=2,
        max_training_examples=1e6,
        window_size_x=30,
        window_size_y=30,
        direc_name=direc_name,
        file_name_save=file_name_save,
        channel_names=['phase'],
        num_of_features=2,
        edge_feature=[1, 0, 0],
        dilation_radius=1,
        border_mode='valid',
        output_mode='sample',
        display=False,
        verbose=True,
        process_std=True)


def train_model_on_training_data():
    batch_size = 1
    n_epoch = 1

    direc_save = os.path.join(DATA_DIR, 'ecoli')
    direc_data = os.path.join(DATA_DIR, 'ecoli')
    dataset = 'ecoli_61x61'

    file_name = os.path.join(direc_data, dataset + '.npz')
    training_data = np.load(file_name)

    print('X.shape: ', training_data['X'].shape)
    print('y.shape: ', training_data['y'].shape)

    model = the_model()

    train_model(
        model=model,
        dataset=dataset,
        optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        batch_size=32,
        n_epoch=1,
        direc_save=direc_save,
        direc_data=direc_data,
        lr_sched=rate_scheduler(lr=0.01, decay=0.99),
        # class_weight=class_weights,
        rotation_range=180,
        flip=True,
        shear=False
    )

if __name__ == '__main__':
    generate_training_data()
    train_model_on_training_data()
