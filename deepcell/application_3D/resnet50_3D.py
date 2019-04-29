from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepcell.keras_application_3D import resnet50_3D

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import tf_export

from deepcell.keras_application_3D import resnet50_3D
from keras_applications import resnet50

@tf_export('keras_application_3D.ResNet50_3D',
           'keras_application_3D.ResNet50_3D')
@keras_modules_injection
def ResNet50_3D(*args, **kwargs):
  return resnet50_3D.ResNet50_3D(*args, **kwargs)


@tf_export('keras.applications.resnet50.decode_predictions')
@keras_modules_injection
def decode_predictions(*args, **kwargs):
  return resnet50.decode_predictions(*args, **kwargs)


@tf_export('keras.applications.resnet50.preprocess_input')
@keras_modules_injection
def preprocess_input(*args, **kwargs):
  return resnet50.preprocess_input(*args, **kwargs)
