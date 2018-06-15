"""
test_generators.py

A simple script to test image generators

Run command:
	python test_generators.py

@author: David Van Valen
"""


from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam

from deepcell import rate_scheduler, train_model_conv as train_model
from deepcell import dilated_bn_feature_net_61x61 as the_model
from deepcell import get_images_from_directory, process_image, get_data
from deepcell import ImageFullyConvDataGenerator

import os
import pathlib
import datetime
import numpy as np
from scipy.misc import imsave

batch_size = 1
n_epoch = 200

dataset = "nuclei_conv_same"

direc_save = "/data/trained_networks/nuclei/"
direc_data = "/data/training_data/training_data_npz/nuclei/"

# Create output ditrectory, if necessary
pathlib.Path( direc_save ).mkdir( parents=True, exist_ok=True )

training_data_file_name = os.path.join(direc_data, dataset + ".npz")
train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode = 'conv')

rotation_range = 0
shear = 0
flip = False

save_to_dir = '/data/training_data_npz/HeLa'
target_format = 'direction'

datagen = ImageFullyConvDataGenerator(
		rotation_range = rotation_range,  # randomly rotate images by 0 to rotation_range degrees
		shear_range = shear, # randomly shear images in the range (radians , -shear_range to shear_range)
		horizontal_flip= flip,  # randomly flip images
		vertical_flip= flip)  # randomly flip images

next(datagen.flow(train_dict, batch_size = 1, save_to_dir = save_to_dir, target_format = target_format))

