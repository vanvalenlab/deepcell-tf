# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Generate Jupyter notebooks for training deep learning models."""

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
                  transform='pixelwise',
                  epochs=10,
                  optimizer='sgd',
                  skips=0,
                  n_frames=5,
                  normalization='std',
                  model_name=None,
                  log_dir=None,
                  export_dir=None,
                  output_dir=os.path.join('scripts', 'generated_notebooks'),
                  **kwargs):
    """Create a training notebook that will step through the training
    process from making an npz file to creating and training a model.

    Args:
        data: zipfile of data to load into npz and train on
        train_type: training method to use, either "sample" or "conv"
        field_size: receptive field of the model, a positive integer
        ndim: dimensionality of the data, either 2 or 3
        transform: transformation to apply to the data
        epochs: number of training epochs
        optimizer: training optimizer (`sgd` or `adam`)
        skips: number of skip connections to use
        n_frames: number of frames to process for 3D data
        normalization: normalization method for ImageNormalization layer
        log_dir: directory to write tensorboard logs
        export_dir: directory to export the model after training
        output_dir: local directory to save the notebook

    Returns:
        notebook_path: path to generated notebook
    """
    if not data:
        raise ValueError('`data` should be a path to the training data.')

    data = str(data).strip()

    train_type = str(train_type).lower()
    if train_type not in {'sample', 'conv'}:
        raise ValueError('`train_type` must be one of "sample" or "conv"')

    try:
        field_size = int(field_size)
        if field_size <= 0:
            raise ValueError
    except:
        raise ValueError('`field_size` must be a positive integer')

    try:
        ndim = int(ndim)
        if ndim not in {2, 3}:
            raise ValueError
    except:
        raise ValueError('`ndim` must be either 2 or 3 for 2D or 3D images')

    if transform is not None:
        transform = str(transform).lower()
        if not transform:
            transform = None

    if transform not in {'pixelwise', 'watershed', None}:
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
        'import random',
        'import shutil',
        'import zipfile',
        '',
        'import numpy as np',
        'from tensorflow.python import keras',
        '',
        'from deepcell.utils.data_utils import make_training_data',
        'from deepcell.utils.data_utils import get_data',
        'from deepcell.utils.io_utils import get_image_sizes',
        'from deepcell.utils.export_utils import export_model',
        'from deepcell.utils.train_utils import rate_scheduler',
        'from deepcell.model_zoo import bn_feature_net_{}D'.format(ndim),
        'from deepcell.model_zoo import bn_feature_net_skip_{}D'.format(ndim),
        'from deepcell.training import train_model_{}'.format(train_type)
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(imports)))

    # Training variables setup
    training_vars = [
        'RESIZE = False',
        'RESHAPE_SIZE = 256',
        '',
        '# filepath constants',
        'ROOT_DIR = os.path.join(os.getcwd(), "output")',
        'DATA_DIR = os.path.join(ROOT_DIR, "data")',
        'MODEL_DIR = os.path.join(ROOT_DIR, "models")',
        'NPZ_DIR = os.path.join(ROOT_DIR, "npz_data")',
        'EXPORT_DIR = "{}"'.format(export_dir),
        'LOG_DIR = "{}"'.format(log_dir),
        'DATA_FILE = "{}"'.format(os.path.splitext(os.path.basename(data))[0]),
        'RAW_PATH = "{}"'.format(data),
        'MODEL_NAME = {}'.format('"{}"'.format(model_name) if model_name else 'None'),
        'FGBG_MODEL_NAME = MODEL_NAME + "_fgbg_"',
        '',
        '# Check for channels_first or channels_last',
        'IS_CHANNELS_FIRST = keras.backend.image_data_format() == "channels_first"',
        'ROW_AXIS = {} if IS_CHANNELS_FIRST else {}'.format(ndim, ndim - 1),
        'COL_AXIS = {} if IS_CHANNELS_FIRST else {}'.format(ndim + 1, ndim),
        'CHANNEL_AXIS = 1 if IS_CHANNELS_FIRST else {}'.format(ndim + 1),
        '',
        'N_FRAMES = {}'.format(n_frames),
        '',
        'for d in (NPZ_DIR, MODEL_DIR, EXPORT_DIR, LOG_DIR, DATA_DIR):',
        '    if not d.startswith("/"):',
        '        continue  # not a local directory, no need to create it',
        '    try:',
        '        os.makedirs(d)',
        '    except OSError as exc:',
        '        if exc.errno != errno.EEXIST:',
        '            raise'
    ]

    cells.append(nbf.v4.new_code_cell('\n'.join(training_vars)))

    # Prepare the zipped data
    data_prep = [
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
    cells.append(nbf.v4.new_code_cell('\n'.join(data_prep)))

    make_data_kwargs = {
        'dimensionality': '{}'.format(ndim),
        'direc_name': 'DATA_DIR',
        'file_name_save': 'os.path.join(NPZ_DIR, DATA_FILE)',
        'training_direcs': 'None',
        'channel_names': [''],
        'raw_image_direc': '"raw"',
        'annotation_direc': '"annotated"',
        'reshape_size': 'RESHAPE_SIZE if RESIZE else None',
    }

    if ndim == 3:
        make_data_kwargs.update({
            'montage_mode': 'True',
            'num_frames': 'None'
        })

    # Make NPZ file from data
    make_data = [
        'make_training_data(',
    ]

    make_data.extend(['    {}={},'.format(k, v) for k, v in make_data_kwargs.items()])
    make_data.extend([
        ')',
        '',
        'assert os.path.isfile(os.path.join(NPZ_DIR, DATA_FILE) + ".npz")'
    ])
    cells.append(nbf.v4.new_code_cell('\n'.join(make_data)))

    if optimizer.lower() == 'sgd':
        opt = 'keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)'
    elif optimizer.lower() == 'adam':
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
    ]
    if ndim == 3:
        load_data.extend([
            '    input_shape = (X.shape[CHANNEL_AXIS], N_FRAMES, size[0], size[1])',
            'else:',
            '    input_shape = (N_FRAMES, size[0], size[1], X.shape[CHANNEL_AXIS])',
        ])
    else:
        load_data.extend([
            '    input_shape = (X.shape[CHANNEL_AXIS], size[0], size[1])',
            'else:',
            '    input_shape = (size[0], size[1], X.shape[CHANNEL_AXIS])',
        ])
    load_data.extend([
        '',
        '# Set up other training parameters',
        'n_epoch = {}'.format(epochs),
        'batch_size = {}'.format(1 if train_type == 'conv' else 32),
        'optimizer = {}'.format(opt),
        'lr_sched = rate_scheduler(lr=0.01, decay=0.99)'
    ])
    cells.append(nbf.v4.new_code_cell('\n'.join(load_data)))

    # Set up model parameters
    if transform == 'pixelwise':
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

    if ndim == 3:
        model_kwargs['n_frames'] = 'N_FRAMES'

    if train_type == 'conv':
        model_kwargs.update({
            'n_skips': skips,
            'last_only': False,
            'multires': False
        })

    # Set up training parameters
    training_kwargs = {
        'model': 'model',
        'dataset': 'os.path.join(DATA_DIR, DATA_FILE + ".npz")',
        'expt': '"{}"'.format(train_type + ('_' + transform if transform else '')),
        'optimizer': 'optimizer',
        'batch_size': 'batch_size',
        'n_epoch': 'n_epoch',
        'model_dir': 'MODEL_DIR',
        'log_dir': 'LOG_DIR',
        'lr_sched': 'lr_sched',
        'rotation_range': 180,
        'flip': True,
        'model_name': 'MODEL_NAME',
        'shear': False,
    }

    if train_type == 'conv' and ndim == 3:
        training_kwargs['frames_per_batch'] = 'N_FRAMES'

    if train_type == 'sample':
        window_size = ((field_size - 1) // 2, (field_size - 1) // 2)
        if ndim == 3:
            window_size = tuple(list(window_size) + [(n_frames - 1) // 2])

        training_kwargs.update({
            'window_size': window_size,
            'balance_classes': kwargs.get('balance_classes', True),
            'max_class_samples': kwargs.get('max_class_samples', int(1e5))
        })

    # FGBG Model
    fgbg_model = [
        '# Instantiate the FGBG separation model',
        'fgbg_model = bn_feature_net{}_{}D('.format(
            '_skip' if train_type == 'conv' else '', ndim),
    ]
    fgbg_model_kwargs = {}
    fgbg_model_kwargs.update(model_kwargs)
    fgbg_model_kwargs['n_features'] = 2
    fgbg_model.extend(['    {}={},'.format(k, v) for k, v in fgbg_model_kwargs.items()])
    fgbg_model.append(')')

    if train_type == 'conv':
        cells.append(nbf.v4.new_code_cell('\n'.join(fgbg_model)))

    fgbg_training_kwargs = {}
    fgbg_training_kwargs.update(training_kwargs)
    fgbg_training_kwargs['expt'] = '"{}_fgbg"'.format(train_type)
    fgbg_training_kwargs['model'] = 'fgbg_model'
    fgbg_training_kwargs['transform'] = '"fgbg"'
    fgbg_training_kwargs['model_name'] = 'FGBG_MODEL_NAME'

    fgbg_training = [
        '# Train the model',
        'fgbg_model = train_model_{}('.format(train_type)
    ]

    fgbg_training.extend(['    {}={},'.format(k, v) for k, v in fgbg_training_kwargs.items()])
    fgbg_training.append(')')

    if train_type == 'conv':
        cells.append(nbf.v4.new_code_cell('\n'.join(fgbg_training)))

    # Instantiate the model
    create_model = [
        '# Instantiate the model',
        'model = bn_feature_net{}_{}D('.format(
            '_skip' if train_type == 'conv' else '', ndim),
    ]
    if train_type == 'conv':
        model_kwargs['fgbg_model'] = 'fgbg_model'
    create_model.extend(['    {}={},'.format(k, v) for k, v in model_kwargs.items()])
    create_model.append(')')
    cells.append(nbf.v4.new_code_cell('\n'.join(create_model)))

    training = [
        '# Train the model',
        'model = train_model_{}('.format(train_type)
    ]

    if transform is not None:
        training_kwargs['transform'] = '"{}"'.format(transform)
    if transform == 'pixelwise':
        training_kwargs['dilation_radius'] = kwargs.get('dilation_radius', 1)
    elif transform == 'watershed':
        training_kwargs['distance_bins'] = kwargs.get('distance_bins', 4)
        training_kwargs['erosion_width'] = kwargs.get('erosion_width', 0)

    training.extend(['    {}={},'.format(k, v) for k, v in training_kwargs.items()])
    training.append(')')
    cells.append(nbf.v4.new_code_cell('\n'.join(training)))

    # Save the weights in a pre-defined filename
    save_weights = [
        '# Save the model weights',
        'weights_path = os.path.join(MODEL_DIR, MODEL_NAME + ".h5")',
        'model.save_weights(weights_path)',
    ]
    if train_type == 'conv':
        save_weights.extend([
            '',
            'fgbg_weights_path = os.path.join(MODEL_DIR, FGBG_MODEL_NAME + ".h5")',
            'fgbg_model.save_weights(fgbg_weights_path)',
        ])
    cells.append(nbf.v4.new_code_cell('\n'.join(save_weights)))

    # Export the trained model
    if train_type == 'sample' or ndim == 3:
        # need to re-initialize the model with new input shape and dilated=False
        dilated_model_kwargs = {}
        dilated_model_kwargs.update(model_kwargs)

        if train_type == 'sample':
            dilated_model_kwargs['dilated'] = 'True'

        create_model = [
            '# Instantiate the dilated model',
            'rand_dir = os.path.join(DATA_DIR, random.choice(os.listdir(DATA_DIR)), "raw")',
        ]

        if ndim == 3:
            create_model.extend([
                '',
                '# 3D datasets have subdirectories',
                'rand_dir = os.path.join(rand_dir, random.choice(os.listdir(rand_dir)))'
            ])

        create_model.append('image_size = get_image_sizes(rand_dir, [""])')

        if ndim == 3:
            shape = '(X.shape[ROW_AXIS - 1], image_size[0], image_size[1], X.shape[CHANNEL_AXIS])'
        else:
            shape = '(image_size[0], image_size[1], X.shape[CHANNEL_AXIS])'

        create_model.extend([
            'dilated_input_shape = {}'.format(shape),
            ''
        ])
        dilated_model_kwargs['input_shape'] = 'dilated_input_shape'

        create_model.append('model = bn_feature_net{}_{}D('.format(
            '' if train_type == 'sample' else '_skip', ndim))

        create_model.extend(['    {}={},'.format(k, v) for k, v in dilated_model_kwargs.items()])
        create_model.append(')')

        cells.append(nbf.v4.new_code_cell('\n'.join(create_model)))

    exports = [
        '# Export the model',
        'export_path = "{}/{}".format(EXPORT_DIR, MODEL_NAME)',
        'model_version = 0',
        'exported = False',
        'while not exported:',
        '    try:',
        '        export_model(model,',
        '            export_path=export_path,',
        '            weights_path=weights_path,',
        '            model_version=model_version,',
        '        )',
        '        exported = True',
        '    except AssertionError:',
        '        model_version += 1',
    ]

    cells.append(nbf.v4.new_code_cell('\n'.join(exports)))

    nb = nbf.v4.new_notebook(cells=cells)

    # Create and write to new ipynb
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    notebook_path = os.path.join(output_dir, 'train_{}.ipynb'.format(st))
    # spaces in filenames can be complicated
    notebook_path = notebook_path.replace(' ', '_')
    nbf.write(nb, notebook_path)
    return notebook_path
