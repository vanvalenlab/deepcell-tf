# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Automated Jupyter Notebook Creation and Server Instantiation
Users can select specific default Jupyter Notebook templates and
enter values for customization. Jupyter Notebook Servers will be
automatically spun up.
@author: andrewqho, willgraf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import shutil
import time

import nbformat as nbf


def make_notebook(data,
                  train_type='conv',
                  field_size=61,
                  dim=2,
                  transform='deepcell',
                  **kwargs):
    """Create a training notebook that will step through the entire
    training process from making an npz file to creating and training
    a deep learning model.
    # Arguments:
        data: zipfile of data to load into npz and train on
        train_type: training method to use, either "sample" or "conv"
        field_size: receptive field of the model, a positive integer
        dim: dimensionality of the data, either 2 or 3
        transform: transformation to apply to the data
    """
    if train_type.lower() not in {'sample', 'conv'}:
        raise ValueError('`train_type` must be one of "sample" or "conv"')
    train_type = train_type.lower()

    if not isinstance(field_size, int) or field_size <=0:
        raise ValueError('`field_size` must be a positive integer')
    
    if dim not in {2, 3}:
        raise ValueError('`dim` must be either 2 or 3 for 2D or 3D images')
    
    if transform not in {None, 'deepcell', 'watershed'}:
        raise ValueError('`transform` got unexpected value', transform)

    # list of cells that will be in the notebook
    cells = []

    # Markdown header
    text = [
        '## Training',
        'A template Jupyter notebook to further train models.',
        'Data is given in compressed form and extracted for training.'
    ]
    cells.append(nbf.v4.new_markdown_cell('\n'.join(text)))

    # Python imports
    imports = [
        'import os',
        'import errno',
        'import zipfile',
        'import numpy as np',
        'from tensorflow.python import keras',
        'from deepcell.utils.data_utils import make_training_data',
        'from deepcell.utils.data_utils import get_data',
        'from deepcell.utils.train_utils import rate_scheduler',
        'from deepcell.model_zoo import bn_feature_net_2D',
        'from deepcell.model_zoo import bn_feature_net_skip_2D',
        'from deepcell.training import train_model_{}'.format(train_type)
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(imports)))

    # Training variables setup
    training_vars = [
        'RESIZE = True',
        'RESHAPE_SIZE = 512',
        '',
        '# filepath constants',
        'DATA_DIR = "./data/data"',
        'MODEL_DIR = "./data/models"',
        'NPZ_DIR = "./data/npz_data"',
        'RESULTS_DIR = "./data/results"',
        'EXPORT_DIR = "./data/exports"',
        'DATA_FILE = "{}"'.format(data),
        '',
        '# Check for channels_first or channels_last',
        'IS_CHANNELS_FIRST = keras.backend.image_data_format() == "channels_first"',
        'ROW_AXIS = {} if IS_CHANNELS_FIRST else {}'.format(dim, dim - 1),
        'COL_AXIS = {} if IS_CHANNELS_FIRST else {}'.format(dim + 1, dim),
        'CHANNEL_AXIS = 1 if IS_CHANNELS_FIRST else {}'.format(dim + 1),
        '',
        'data_zip = zipfile.ZipFile(DATA_FILE)',
        'data_zip.extractall(DATA_DIR)',
        'data_zip.close()',
        '',
        'for d in (NPZ_DIR, MODEL_DIR, RESULTS_DIR):',
        '    os.makedirs(d)'
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(training_vars)))

    # Make NPZ file from data
    make_data = [
        'make_training_data(',
        '    dimensionality={dim},  # 2D or 3D data'.format(dim=dim),
        '    direc_name=DATA_DIR,',
        '    file_name_save=os.path.join(NPZ_DIR, DATA_FILE),',
        '    training_direcs=None',
        '    channel_names=[""],  # matches image files as wildcard',
        '    raw_image_direc="raw"',
        '    annotation_direc="annotated",  # directory name of label data',
        '    reshape_size=RESHAPE_SIZE if RESIZE else None)',
        '',
        'if os.path.isfile(os.path.join(NPZ_DIR, DATA_FILE) + ".npz"):',
        '    print("Data Saved to", os.path.join(NPZ_DIR, DATA_FILE) + ".npz")',
        'else:',
        '    raise Exception("Uh Oh!  Your data file did not save properly :(")'
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(make_data)))

    # Load data from NPZ
    load_data = [
        '# Load the training data from NPZ into a numpy array',
        'training_data = np.load(os.path.join(NPZ_DIR, DATA_FILE + ".npz"))',
        '',
        'X, y = training_data["X"], training_data["y"]',
        'print("X.shape: {} & y.shape: {}".format(X.shape, y.shape))',
        '',
        '# save the size of the input data for input_shape model parameter',
        'size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[ROW_AXIS:COL_AXIS + 1]',
        'if IS_CHANNELS_FIRST:',
        '    input_shape = (X.shape[CHANNEL_AXIS], size[0], size[1])',
        'else:',
        '    input_shape = (size[0], size[1], X.shape[CHANNEL_AXIS])',
        '',
        '# Set up other training parameters',
        'n_epoch = 10',
        'batch_size = {}'.format(1 if train_type == 'conv' else 32),
        'optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)',
        'lr_sched = rate_scheduler(lr=0.01, decay=0.99)'
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(load_data)))

    # Instantiate the model
    model_class = 'bn_feature_net{}_{}D'.format(
        '_skip' if train_type == 'conv' else '', dim)
    
    if transform == 'deepcell':
        n_features = 4
    elif transform == 'watershed':
        n_features = kwargs.get('distance_bins', 4)
    else:  # transform is None
        n_features = 2
    
    model_kwargs = {
        'receptive_field': field_size,
        'n_channels': 'X.shape[CHANNEL_AXIS]',
        'input_shape': 'input_shape',
        'norm_method': 'median' if dim == 2 else 'whole_image',
        'reg': 1e-5,
        'n_conv_filters': 32,
        'n_dense_filters': 128,
        'n_features': n_features
    }

    if train_type == 'conv':
        model_kwargs.update({
            'n_skips': 3,
            'last_only': False,
            'multires': False
        })

    create_model = [
        '# Instantiate the model',
        'model = {}('.format(model_class),
    ]

    create_model.extend(['    {}={},'.format(k, v) for k, v in model_kwargs.items()])
    create_model.append(')')
    cells.append(nbf.v4.new_code_cell('\n'.join(create_model)))

    # Call training function
    training_kwargs = {
        'model': 'model',
        'dataset': 'DATA_FILE',
        'expt': '"{}"'.format(train_type + ('_' + transform if transform else '')),
        'optimizer': 'optimizer',
        'batch_size': 'batch_size',
        'n_epoch': 'n_epoch',
        'direc_save': 'os.path.join(MODEL_DIR, PREFIX)',
        'direc_data': 'os.path.join(NPZ_DIR, PREFIX)',
        'lr_sched': 'lr_sched',
        'rotation_range': 180,
        'flip': True,
        'shear': False
    }

    if train_type == 'sample':
        training_kwargs.update({
            'window_size': (field_size - 1 // 2, field_size - 1 // 2),
            'balance_classes': kwargs.get('balance_classes', True),
        })
        if 'max_class_samples' in kwargs:
            training_kwargs['max_class_samples'] = kwargs.get('max_class_samples')

    if transform is not None:
        training_kwargs['transform'] = '"{}"'.format(transform)
    if transform == 'deepcell':
        training_kwargs['dilation_radius'] = kwargs.get('dilation_radius', 1)
    elif transform == 'watershed':
        training_kwargs['distance_bins'] = kwargs.get('dilation_radius', 1)
        training_kwargs['erosion_width'] = kwargs.get('erosion_width', 0)

    training = ['train_model_{}('.format(train_type)]
    training.extend(['    {}={},'.format(k, v) for k, v in training_kwargs.items()])
    training.append(')')

    nb = nbf.v4.new_notebook(cells=cells)
    nb['cells'].extend([
        nbf.v4.new_code_cell('\n'.join(training))
    ])

    # Create and write to new ipynb
    nbf.write(nb, 'train.ipynb')

    # Move data file to "notebook" directory
    # shutil.move(data, os.path.join('notebooks', os.path.basename(data)))

    # # Change CWD to "notebook" directory
    # os.chdir(os.path.join('.', 'notebooks'))

    # # Create directory for notebook and data
    # ts = '{}'.format(time.time()).encode('utf-8')
    # hashed_directory = 'train_{}'.format(hashlib.md5(ts).hexdigest())
    # os.mkdir(hashed_directory)

    # # Move data file to new directory
    # shutil.move(data, os.path.join(hashed_directory, os.path.basename(data)))

    # # Change CWD to new directory
    # os.chdir(hashed_directory)

    # # Create data directory
    # os.mkdir('data')

    # # Move data file to data directory
    # shutil.move(data, os.path.join('data', os.path.basename(data)))
    
    os.system('jupyter notebook')
