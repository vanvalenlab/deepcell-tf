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

import os
import errno
import time
import datetime

import nbformat as nbf


def make_notebook(data,
                  train_type='conv',
                  field_size=61,
                  ndim=2,
                  transform='deepcell',
                  epochs=10,
                  optimizer='sgd',
                  skips=0,
                  normalization='std',
                  model_name=None,
                  log_dir=None,
                  export_dir=None,
                  output_dir=os.path.join('scripts', 'generated_notebooks'),
                  **kwargs):
    """Create a training notebook that will step through the training
    process from making an npz file to creating and training a model.
    # Arguments:
        data: zipfile of data to load into npz and train on
        train_type: training method to use, either "sample" or "conv"
        field_size: receptive field of the model, a positive integer
        ndim: dimensionality of the data, either 2 or 3
        transform: transformation to apply to the data
        epochs: number of training epochs
        optimizer: training optimizer (`sgd` or `adam`)
        skips: number of skip connections to use
        normalization: normalization method for ImageNormalization layer
        log_dir: directory to write tensorboard logs
        export_dir: directory to export the model after training
        output_dir: local directory to save the notebook
    # Returns:
        notebook_path: path to generated notebook
    """
    if not data:
        raise ValueError('`data` should be a path to the training data.')
    data = str(data).strip()

    train_type = str(train_type).lower()
    if train_type not in {'sample', 'conv'}:
        raise ValueError('`train_type` must be one of "sample" or "conv"')

    if not isinstance(field_size, int) or field_size <= 0:
        raise ValueError('`field_size` must be a positive integer')

    if ndim not in {2, 3}:
        raise ValueError('`ndim` must be either 2 or 3 for 2D or 3D images')

    if transform is not None:
        transform = str(transform).lower()
        if transform not in {'deepcell', 'watershed'}:
            raise ValueError('`transform` got unexpected value', transform)

    if normalization is not None:
        normalization = str(normalization).lower()
        if normalization not in {'std', 'max', 'whole_image', 'median'}:
            raise ValueError('`normalization` got unexpected value', transform)

    if export_dir is None:
        export_dir = 'os.path.join(ROOT_DIR, "exports")'
    if log_dir is None:
        log_dir = 'os.path.join(ROOT_DIR, "tensorboard_logs")'

    try:
        os.makedirs(output_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

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
        'import shutil',
        'import zipfile',
        '',
        'import numpy as np',
        'from tensorflow.python import keras',
        '',
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
        'RESIZE = False',
        'RESHAPE_SIZE = 256',
        '',
        '# filepath constants',
        'ROOT_DIR = "{}"'.format(os.path.join('.', 'output')),
        'DATA_DIR = os.path.join(ROOT_DIR, "data")',
        'MODEL_DIR = os.path.join(ROOT_DIR, "models")',
        'NPZ_DIR = os.path.join(ROOT_DIR, "npz_data")',
        'EXPORT_DIR = "{}"'.format(export_dir),
        'LOG_DIR = "{}"'.format(log_dir),
        'DATA_FILE = "{}"'.format(os.path.splitext(os.path.basename(data))[0]),
        'RAW_PATH = "{}"'.format(data),
        '',
        '# Check for channels_first or channels_last',
        'IS_CHANNELS_FIRST = keras.backend.image_data_format() == "channels_first"',
        'ROW_AXIS = {} if IS_CHANNELS_FIRST else {}'.format(ndim, ndim - 1),
        'COL_AXIS = {} if IS_CHANNELS_FIRST else {}'.format(ndim + 1, ndim),
        'CHANNEL_AXIS = 1 if IS_CHANNELS_FIRST else {}'.format(ndim + 1),
        '',
        'for d in (NPZ_DIR, MODEL_DIR, EXPORT_DIR, LOG_DIR, DATA_DIR):',
        '    if not d.startswith("/"):',
        '        continue  # not a local directory, no need to create it',
        '    try:',
        '        os.makedirs(d)',
        '    except OSError as exc:',
        '        if exc.errno != errno.EEXIST:',
        '            raise',
        '',
        'if zipfile.is_zipfile(RAW_PATH):',
        '    archive = zipfile.ZipFile(RAW_PATH)',
        '    for info in archive.infolist():',
        '        # skip OSX archiving artifacts',
        '        if "__MACOSX" in info.filename or ".DStore" in info.filename:',
        '            continue',
        '',
        '        archive.extract(info, path=DATA_DIR)',
        '',
        '    archive.close()',
        '',
        '    # If the zip file did not have a top level directory, create one.',
        '    children = os.listdir(DATA_DIR)',
        '    if len(children) > 1:',
        '        top_level = os.path.join(DATA_DIR, os.path.basename(RAW_PATH))',
        '        os.makedirs(top_level)',
        '        for child in children:',
        '            shutil.move(os.path.join(DATA_DIR, child), top_level)',
        '',
        '        DATA_DIR = top_level',
        '    else:',
        '        DATA_DIR = os.path.join(DATA_DIR, children[0])'
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(training_vars)))

    # Make NPZ file from data
    make_data = [
        'try:',
        '    make_training_data(',
        '        dimensionality={ndim},  # 2D or 3D data'.format(ndim=ndim),
        '        direc_name=DATA_DIR,',
        '        file_name_save=os.path.join(NPZ_DIR, DATA_FILE),',
        '        training_direcs=None,',
        '        channel_names=[""],  # matches image files as wildcard',
        '        raw_image_direc="raw",',
        '        annotation_direc="annotated",  # directory name of label data',
        '        reshape_size=RESHAPE_SIZE if RESIZE else None)',
        'except Exception as err:',
        '    raise Exception("Could not create training data due to error: {}".format(err))',
        '',
        'if os.path.isfile(os.path.join(NPZ_DIR, DATA_FILE) + ".npz"):',
        '    print("Data Saved to", os.path.join(NPZ_DIR, DATA_FILE) + ".npz")',
        'else:',
        '    raise Exception("Uh Oh!  Your data file did not save properly :(")'
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(make_data)))

    if optimizer.lower() == 'sgd':
        opt = 'keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)'
    elif optimizer.lower == 'adam':
        opt = 'keras.optimizers.Adam(lr=0.01, epsilon=None, decay=1e-6)'
    else:
        raise ValueError('Invalid optimizer value: `{}`'.format(optimizer))

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
        'n_epoch = {}'.format(epochs),
        'batch_size = {}'.format(1 if train_type == 'conv' else 32),
        'optimizer = {}'.format(opt),
        'lr_sched = rate_scheduler(lr=0.01, decay=0.99)'
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(load_data)))

    # Instantiate the model
    create_model = [
        '# Instantiate the model',
        'model = bn_feature_net{}_{}D('.format(
            '_skip' if train_type == 'conv' else '', ndim),
    ]

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
        'norm_method': '"{}"'.format(normalization),
        'reg': 1e-5,
        'n_conv_filters': 32,
        'n_dense_filters': 128,
        'n_features': n_features
    }

    if train_type == 'conv':
        model_kwargs.update({
            'n_skips': skips,
            'last_only': False,
            'multires': False
        })

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
        'direc_save': 'MODEL_DIR',
        'direc_data': 'NPZ_DIR',
        'log_dir': 'LOG_DIR',
        'lr_sched': 'lr_sched',
        'rotation_range': 180,
        'flip': True,
        'shear': False,
    }

    if model_name is not None:
        training_kwargs['model_name'] = '"{}"'.format(model_name)

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
        training_kwargs['distance_bins'] = kwargs.get('distance_bins', 4)
        training_kwargs['erosion_width'] = kwargs.get('erosion_width', 0)

    training = [
        '# Train the model',
        'model = train_model_{}('.format(train_type)
    ]
    training.extend(['    {}={},'.format(k, v) for k, v in training_kwargs.items()])
    training.append(')')
    cells.append(nbf.v4.new_code_cell('\n'.join(training)))


    nb = nbf.v4.new_notebook(cells=cells)

    # Create and write to new ipynb
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    notebook_path = os.path.join(output_dir, 'train_{}.ipynb'.format(st))
    # spaces in filenames can be complicated
    notebook_path = notebook_path.replace(' ', '_')
    nbf.write(nb, notebook_path)
    return notebook_path
