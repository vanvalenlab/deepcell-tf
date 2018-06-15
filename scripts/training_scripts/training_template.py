"""
training_template.py

Train a simple deep CNN on a dataset.

Run command:
	python training_template.py

@author: David Van Valen
"""


from tensorflow.python.keras.optimizers import SGD, RMSprop

from deepcell import rate_scheduler, train_model_sample as train_model
from deepcell import bn_feature_net_61x61 as the_model

import os
import pathlib
import datetime
import numpy as np

batch_size = 256
n_epoch = 30

dataset = "HeLa_all_61x61"
expt = "bn_feature_net_61x61"

direc_save = "/data/trained_networks/HeLa/"
direc_data = "/data/training_data_npz/HeLa/"

# Create output ditrectory, if necessary
pathlib.Path( direc_save ).mkdir( parents=True, exist_ok=True )

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.99)

class_weights = {0:1, 1:1, 2:1}

for iterate in range(1):

	model = the_model(n_channels = 2, n_features = 3, reg = 1e-5)

	train_model(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, class_weight = class_weights,
		rotation_range = 180, flip = True, shear = False)
