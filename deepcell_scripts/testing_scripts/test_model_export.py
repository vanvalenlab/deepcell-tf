"""
test_export.py

A simple script to test exporting models for
tensorflow serving

Run command:
	python test_model_export.py

@author: David Van Valen
"""

from __future__ import print_function
from __future__ import division

import os
import warnings

from deepcell import export_model
from deepcell.model_zoo import dilated_feature_net_61x61 as model_fn

weights_path = '/models/2018-05-25_ecoli_kc_polaris_61x61__0.h5'
export_path = '/exported_models'

export_model(keras_model, export_path, weights_path = weights_path)



