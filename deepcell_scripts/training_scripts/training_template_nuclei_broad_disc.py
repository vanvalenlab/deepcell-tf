"""
training_template.py

Train a simple deep CNN on a dataset.

Run command:
	python training_template.py

@author: David Van Valen
"""


from tensorflow.python.keras.optimizers import SGD, RMSprop

from deepcell import rate_scheduler, train_model_disc as train_model
from deepcell import bn_dense_feature_net as the_model

import os
import pathlib
import datetime
import numpy as np

batch_size = 1
n_epoch = 100

dataset = "nuclei_broad_same_disc_61x61"
expt = "bn_dense_feature_net"

direc_save = "/data/trained_networks/nuclei_broad/"
direc_data = "/data/training_data_npz/nuclei_broad/"

# Create output ditrectory, if necessary
pathlib.Path( direc_save ).mkdir( parents=True, exist_ok=True )

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.99)

file_name = os.path.join(direc_data, dataset + ".npz")
training_data = np.load(file_name)
class_weights = training_data["class_weights"]

for iterate in range(1):

	model = the_model(batch_shape = (1, 512, 512, 1), n_features = 16, reg = 1e-5, softmax = False, location = True, permute = True)

	train_model(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, class_weight = class_weights,
		rotation_range = 180, flip = True, shear = False)


