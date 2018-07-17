
#Backbone usage : python retinanet_custombackbone_train.py --no-weights --image-min-side 360 --image-max-side 426 --random-transform --backbone shvm --steps=1000 --epochs=10 --gpu 0 --tensorboard-dir logs csv ./annotation.csv ./classes.csv

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import keras
from keras.utils import get_file
import keras_resnet
import keras_resnet.models

from keras_retinanet.models import retinanet
from keras_retinanet.models import Backbone
from keras_retinanet.utils.image import preprocess_image
'''
First one needs to import the Backbone class from the custom_backbone defination"
'''
class shvmBackbone(Backbone):

    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(shvmBackbone, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return shvm_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        '''
        If your custom model ha spretrained weights on imagenet you could write the method to download it here
        else return None and use the --no-weights flag while training.
        '''
        return None

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['shvm']
        backbone = self.backbone.split('_')[0]   #To allow diffrent versions of same backbone


    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


'''Write your own nueral network here, its mandatory to make it in a function which expects the keras tensor 'inputs' instead of numpy array (batch,x,y,channel) as an argument for graph consitency.'''
def tempnetwork(inputs):
    '''
    Retinanet uses feautre pyramid networks(FPN) which combines features from diffrent scales in the nueral
    network to obtain a better understanding of the image.

    Retinanet requires 3 features from the backbone called C3 , C4 and C5 , now the scaling here is as follows:
    If the model inputs are of the form -: Inputs : (x , x , 3)
    then  Cn = (x//(2^n) , x//(2^n) , None)
    Here a//b denotes rounding the float (a/b) to the largest integer.
    '''
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='shvm1')(conv4)
    #I would custom name the 3 layers I would require form the backbone
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='shvm2')(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

    conv6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='shvm3')(conv6)
    drop6 = Dropout(0.5)(conv6)
    

    model = Model(inputs = inputs, outputs = conv6)
    
    #model.summary()
    return model
    
    
    
    
    
def shvm_retinanet(num_classes, backbone='shvm', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using the custom backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Our custom backbone.
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone like it could be used to freeze the training of the backbone.

    Returns
        RetinaNet model with a custom backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    modeltest = tempnetwork(inputs)               # Call the custom model
    layer_names = ['shvm1', 'shvm2', 'shvm3']     # Make an array of the names of the layers we want
    layer_outputs = [modeltest.get_layer(name).output for name in layer_names] # Get the required layers
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name='shvm')

    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone.outputs, **kwargs)

