"""
running_template_MIBI.py
Run a trained CNN on multiple MIBI datasets.

Run command:
	python running_template_MIBI.py

@author: David Van Valen
"""

import h5py
import tifffile as tiff

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes
from model_zoo import dilated_bn_feature_net_61x61 as nuclear_fn

import os
import numpy as np


"""
Load data
"""

base_direc = '/home/vanvalen/Data/MIBI/AllData/'
output_direc = '/home/vanvalen/Data/MIBI/Output/'
list_of_dirs = os.listdir(base_direc)

for sub_direc in ["Point45", "Point46", "Point47", "Point48"]:
	direc_name = os.path.join(base_direc, sub_direc)
	output_name = os.path.join(output_direc, sub_direc)
	if os.path.isdir(output_name) is False:
		os.mkdir(output_name)

	print direc_name, output_name

	data_location = direc_name
	nuclear_location = output_name

	nuclear_channel_names = ["dsDNA", "H3K9ac", "H3K27me3"]

	trained_network_nuclear_directory = "/home/vanvalen/Data/MIBI/trained_networks"

	nuclear_prefix = "2017-10-16_Samir_1e6_61x61_bn_feature_net_61x61_"

	win = 30

	image_size_x, image_size_y = get_image_sizes(data_location, nuclear_channel_names)

	"""
	Define model
	"""

	list_of_nuclear_weights = []
	for j in xrange(1):
		nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix + str(0) + ".h5")
		list_of_nuclear_weights += [nuclear_weights]

	"""
	Run model on directory
	"""

	nuclear_predictions = run_models_on_directory(data_location, nuclear_channel_names, nuclear_location, n_features = 3, model_fn = nuclear_fn, 
		list_of_weights = list_of_nuclear_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
		win_x = win, win_y = win, std = True, split = True)



