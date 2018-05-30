"""
running_template.py
Run a trained CNN on a dataset.

Run command:
    python training_template.py

@author: David Van Valen
"""

import os

from deepcell import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes
from deepcell import bn_feature_net_61x61


channel_names = ['phase']

data_location = '/data/ecoli_kc/set1'

image_size_x, image_size_y = get_image_sizes(data_location, channel_names)

"""
Define model
"""

model_dir = '/data'
model_name = '2018-05-21_ecoli_61x61__0.h5'
weights = os.path.join(model_dir, model_name)

"""
Run model on directory
"""

predictions = run_models_on_directory(
    data_location, channel_names, '/data/ecoli_kc_results',
    n_features=3,
    model_fn=bn_feature_net_61x61,
    list_of_weights=[weights],
    image_size_x=image_size_x,
    image_size_y=image_size_y,
    win_x=30,
    win_y=30,
    std=True,
    split=True)
