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


from deepcell import make_training_data

# Define maximum number of training examples
window_size = 30

# Load data
direc_name = '/data/data/cells/unspecified_nuclear_data/nuclear_movie'
output_directory = '/data/npz_data/cells/unspecified_nuclear_data/nuclear_movie/'
file_name_save = os.path.join( output_directory, 'nuclear_movie_same.npz')
training_direcs = ["set1", "set2"]
channel_names = ["DAPI"]

# Create output ditrectory, if necessary
pathlib.Path( output_directory ).mkdir( parents=True, exist_ok=True )

# Create the training data
make_training_data(window_size_x = 30, window_size_y = 30,
	direc_name = direc_name,
    montage_mode=False,
	file_name_save = file_name_save,
	training_direcs = training_direcs,
	channel_names = channel_names,
	dimensionality = 3,
	annotation_name = "corrected",
	raw_image_direc = "RawImages",
	annotation_direc = "Annotation",
	border_mode = "same",
	num_frames = 60,
	reshape_size = None,
	display = False,
	num_of_frames_to_display = 5,
	verbose = True)
