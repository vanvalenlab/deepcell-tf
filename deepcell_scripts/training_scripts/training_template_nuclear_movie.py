"""
training_template.py

Train a simple deep CNN on a dataset in a fully convolutional fashion.

Run command:
	python training_template_fully_conv.py

@author: David Van Valen
"""


from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam

from deepcell import rate_scheduler, train_model_movie as train_model
from deepcell import bn_dense_feature_net_3D as the_model
from deepcell import get_images_from_directory

import os
import pathlib
import datetime
import numpy as np
from scipy.misc import imsave

batch_size = 1
n_epoch = 50

dataset = "nuclear_movie_disc_same"
expt = "bn_dense_net_3D"

direc_save = "/data/trained_networks/nuclear_movie/"
direc_data = "/data/training_data_npz/nuclear_movie/"

# Create output ditrectory, if necessary
pathlib.Path( direc_save ).mkdir( parents=True, exist_ok=True )

optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 1e-2, decay = 0.99)

file_name = os.path.join(direc_data, dataset + ".npz")
training_data = np.load(file_name)

for iterate in range(1):

	model = the_model(batch_shape = (1,1,5,256,256), n_features = 3, reg = 1e-5, location = False, permute = True, softmax = False)

	trained_model = train_model(model = model, dataset = dataset, optimizer = optimizer,
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, number_of_frames = 5,
		lr_sched = lr_sched, rotation_range = 180, flip = True, shear = False)
