"""
make_training_data.py

Executing functions for creating npz files containing the training data
Functions will create training data for either
	- Patchwise sampling
	- Fully convolutional training of single image conv-nets
	- Fully convolutional training of movie conv-nets

Files should be plased in training directories with each separate
dataset getting its own folder

@author: David Van Valen
"""

"""
Import packages
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import glob
import os
import pathlib
import skimage as sk
import scipy as sp
from scipy import ndimage
from skimage import feature
from sklearn.utils import class_weight
from deepcell import get_image
from deepcell import format_coord as cf
from skimage import morphology as morph
import matplotlib.pyplot as plt
from skimage.transform import resize


from deepcell import make_training_data_2d as make_training_data

# Define maximum number of training examples
max_training_examples = 1e6
window_size = 30

# Load data
direc_name = '/data/training_data/HeLa_joint/'
output_directory = '/data/training_data/training_data_npz/HeLa/'
file_name_save = os.path.join( output_directory, 'HeLa_joint_conv_same_61x61.npz')
training_direcs = ["set1", "set2", "set3", "set4", "set5"]
channel_names = ["phase", "nuclear"]

# Create output ditrectory, if necessary
pathlib.Path( output_directory ).mkdir( parents=True, exist_ok=True )

# Specify the number of feature masks that are present
num_of_features = 2

# Specify which feature is the edge feature
edge_feature = [1,0,0]

# Create the training data
make_training_data(max_training_examples = max_training_examples, window_size_x = window_size, window_size_y = window_size,
		direc_name = direc_name,
		file_name_save = file_name_save,
		training_direcs = training_direcs,
		channel_names = channel_names,
		edge_feature = edge_feature,
		dilation_radius = 1,
		border_mode = "same",
		output_mode = "conv",
		reshape_size = 512,
		display = False,
		max_plotted = 5,
		verbose = True)
