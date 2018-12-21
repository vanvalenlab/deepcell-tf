# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities training neural nets"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.models import Model
from tensorflow.python.client import device_lib


def rate_scheduler(lr=.001, decay=0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn


def count_gpus():
    devices = device_lib.list_local_devices()
    gpus = [d for d in devices if d.name.lower().startswith('/device:gpu')]
    return len(gpus)


class MultiGpuModel(Model):
    """Wrapper Model class to enable multi-gpu saving/loading"""
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        """Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model."""
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(MultiGpuModel, self).__getattribute__(attrname)
