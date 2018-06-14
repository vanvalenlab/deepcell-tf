"""
train_utils.py

Functions to help with training neural nets

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.python.keras import activations

def axis_softmax(x, axis=1):
    return activations.softmax(x, axis=axis)

def rate_scheduler(lr=.001, decay=0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn
