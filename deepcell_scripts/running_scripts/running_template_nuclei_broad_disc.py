"""
running_template.py
Run a trained CNN on a dataset.

Run command:
	python training_template.py

@author: David Van Valen
"""

import h5py
import tifffile as tiff

from deepcell import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes 
from deepcell import bn_dense_feature_net as nuclear_fn
from tensorflow.python.keras import backend as K

import os
import numpy as np


"""
Load data
"""

direc_name = '/data/testing_data/nuclei_broad/set1'
data_location = os.path.join(direc_name, 'RawImages')
nuclear_location = os.path.join(direc_name, 'Nuclear')
mask_location = os.path.join(direc_name, 'Masks')

nuclear_channel_names = ['nuclear']

trained_network_nuclear_directory = "/data/trained_networks/nuclei_broad/"

nuclear_prefix = "2018-02-20_nuclei_broad_same_disc_61x61_bn_dense_feature_net_"

win_nuclear = 30

image_size_x, image_size_y = get_image_sizes(data_location, nuclear_channel_names)

"""
Define model
"""

list_of_nuclear_weights = []
for j in xrange(1):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix + str(j) + ".h5")
	list_of_nuclear_weights += [nuclear_weights]

print list_of_nuclear_weights

"""
Run model on directory
"""

nuclear_predictions = run_models_on_directory(data_location, nuclear_channel_names, nuclear_location, model_fn = nuclear_fn, 
	n_features = 16, list_of_weights = list_of_nuclear_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_nuclear, win_y = win_nuclear, std = False, split = False)


