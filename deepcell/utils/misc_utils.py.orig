# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""Miscellaneous utility functions"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re

from tensorflow.python.keras import applications
from tensorflow.python.keras.models import Model

from deepcell import homeapplications


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_pyramid_layer_outputs(backbone, inputs, **kwargs):
    _backbone = str(backbone).lower()

    vgg_backbones = {'vgg16', 'vgg19'}
    densenet_backbones = {'densenet121', 'densenet169', 'densenet201'}
    mobilenet_backbones = {'mobilenet', 'mobilenetv2', 'mobilenet_v2'}
    resnet_backbones = {'resnet50'}
    nasnet_backbones = {'nasnet_large', 'nasnet_mobile'}

    ## 3D ADDING
    resnet_3D_backbones = {'resnet50_3d', 'resnet50_3D_gne'}
<<<<<<< HEAD
=======
    print(resnet_3D_backbones)
>>>>>>> 2397d19954b6ab6310ff6547d183f88fa51bba40

    if _backbone in vgg_backbones:
        layer_names = ['block3_pool', 'block4_pool', 'block5_pool']
        if _backbone == 'vgg16':
            model = applications.VGG16(**kwargs)
        else:
            model = applications.VGG19(**kwargs)
        return [model.get_layer(n).output for n in layer_names]

    elif _backbone in densenet_backbones:
        if _backbone == 'densenet121':
            model = applications.DenseNet121(**kwargs)
            blocks = [6, 12, 24, 16]
        elif _backbone == 'densenet169':
            model = applications.DenseNet169(**kwargs)
            blocks = [6, 12, 32, 32]
        elif _backbone == 'densenet201':
            model = applications.DenseNet201(**kwargs)
            blocks = [6, 12, 48, 32]
        layer_outputs = []
        for idx, block_num in enumerate(blocks):
            name = 'conv{}_block{}_concat'.format(idx + 2, block_num)
            layer_outputs.append(model.get_layer(name=name).output)
        # create the densenet backbone
        model = Model(inputs=inputs, outputs=layer_outputs[1:], name=model.name)
        return model.outputs

    elif _backbone in resnet_backbones:
        model = applications.ResNet50(**kwargs)
        layer_names = ['res3d_branch2c', 'res4f_branch2c', 'res5c_branch2c']
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        model = Model(inputs=inputs, outputs=layer_outputs, name=model.name)
        return model.outputs

    ## 3D ADDING
    elif _backbone in resnet_3D_backbones:
        model = homeapplications.resnet50_3D(**kwargs)
        layer_names = ['res3d_branch2c', 'res4f_branch2c', 'res5c_branch2c']
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        model = Model(inputs=inputs, outputs=layer_outputs, name=model.name)
        return model.outputs

    elif _backbone in mobilenet_backbones:
        alpha = kwargs.get('alpha', 1.0)
        if _backbone.endswith('v2'):
            model = applications.MobileNetV2(alpha=alpha, **kwargs)
            block_ids = (12, 15, 16)
            layer_names = ['block_%s_depthwise_relu' % i for i in block_ids]
        else:
            model = applications.MobileNet(alpha=alpha, **kwargs)
            block_ids = (5, 11, 13)
            layer_names = ['conv_pw_%s_relu' % i for i in block_ids]
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        model = Model(inputs=inputs, outputs=layer_outputs, name=model.name)
        return model.outputs

    elif _backbone in nasnet_backbones:
        if _backbone.endswith('large'):
            model = applications.NASNetLarge(**kwargs)
            block_ids = [5, 12, 18]
        else:
            model = applications.NASNetMobile(**kwargs)
            block_ids = [3, 8, 12]
        layer_names = ['normal_conv_1_%s' % i for i in block_ids]
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        model = Model(inputs=inputs, outputs=layer_outputs, name=model.name)
        return model.outputs

    else:
        backbones = list(densenet_backbones + resnet_backbones + vgg_backbones + resnet_3D_backbones)
        raise ValueError('Invalid value for `backbone`. Must be one of: %s' %
                         ', '.join(backbones))
