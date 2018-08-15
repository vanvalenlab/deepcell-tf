"""
model_zoo.py

Assortment of CNN architectures for single cell segmentation

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Conv3D, ConvLSTM2D, LSTM
from tensorflow.python.keras.layers import Add, Input, Concatenate, Lambda
from tensorflow.python.keras.layers import MaxPool2D, MaxPool3D, AvgPool2D, UpSampling2D
from tensorflow.python.keras.layers import Flatten, Dense, Dropout, Reshape
from tensorflow.python.keras.layers import Activation, Softmax
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.regularizers import l2

from .layers import Resize
from .layers import DilatedMaxPool2D, DilatedMaxPool3D
from .layers import TensorProd2D, TensorProd3D
from .layers import Location, Location3D
from .layers import ImageNormalization2D, ImageNormalization3D


"""
Batch normalized conv-nets
"""


def bn_feature_net_21x21(n_features=3, n_channels=1, reg=1e-5, init='he_normal', norm_method='std'):
    print('Using feature net 21x21 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        input_shape = (n_channels, 21, 21)
    else:
        channel_axis = -1
        input_shape = (21, 21, n_channels)

    model = Sequential()
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=21, input_shape=input_shape))
    model.add(Conv2D(32, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(200, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Flatten())

    model.add(Softmax(axis=channel_axis))

    return model


def dilated_bn_feature_net_21x21(input_shape=(2, 1080, 1280), batch_size=None, n_features=3, reg=1e-5, init='he_normal', weights_path=None, norm_method='std'):
    print('Using dilated feature net 21x21 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    model = Sequential()
    d = 1
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=21, input_shape=input_shape))
    model.add(Conv2D(32, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', batch_size=batch_size, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))

    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(200, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

    model.add(Softmax(axis=channel_axis))

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def bn_feature_net_31x31(n_features=3, n_channels=1, reg=1e-5, init='he_normal', norm_method='std'):
    print('Using feature net 31x31 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        input_shape = (n_channels, 31, 31)
    else:
        channel_axis = -1
        input_shape = (31, 31, n_channels)

    model = Sequential()
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=31, input_shape=input_shape))
    model.add(Conv2D(32, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(200, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Flatten())

    model.add(Softmax(axis=channel_axis))

    return model


def dilated_bn_feature_net_31x31(input_shape=(2, 1080, 1280), n_features=3, n_channels=1, reg=1e-5, init='he_normal', weights_path=None, norm_method='std'):
    print('Using dilated feature net 31x31 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    model = Sequential()
    d = 1
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=31, input_shape=input_shape))
    model.add(Conv2D(32, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(128, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(200, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

    model.add(Softmax(axis=channel_axis))

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def bn_feature_net_61x61(n_features=3, n_channels=1, reg=1e-5, init='he_normal', norm_method='std'):
    print('Using feature net 61x61 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    input_shape = (n_channels, 61, 61) if channel_axis == 1 else (61, 61, n_channels)
    model = Sequential()
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=61, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(200, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Flatten())

    model.add(Activation('softmax'))

    return model


def dilated_bn_feature_net_61x61(input_shape=(2, 1080, 1280), batch_size=None, n_features=3, reg=1e-5, init='he_normal', weights_path=None, norm_method='std'):
    print('Using dilated feature net 61x61 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    model = Sequential()
    d = 1
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=61, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', batch_size=batch_size, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(200, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Softmax(axis=channel_axis))

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def bn_feature_net_81x81(n_features=3, n_channels=1, reg=1e-5, init='he_normal', norm_method='std'):
    print('Using feature net 81x81 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    model = Sequential()
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=81, input_shape=(n_channels, 81, 81)))
    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(200, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Flatten())

    model.add(Softmax(axis=channel_axis))

    return model


def dilated_bn_feature_net_81x81(input_shape=(2, 1080, 1280), n_features=3, reg=1e-5, init='he_normal', weights_path=None, norm_method='std'):
    print('Using dilated feature net 81x81 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    model = Sequential()
    d = 1
    model.add(ImageNormalization2D(norm_method=norm_method, filter_size=81, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool2D(pool_size=(2, 2)))
    d *= 2

    model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(200, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

    model.add(Softmax(axis=channel_axis))

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


"""
Multi-resolution batch normalized conv-nets
"""


def bn_multires_feature_net_61x61(n_features=3, n_channels=1, reg=1e-5, init='he_normal', norm_method='std'):
    print('Using multi-resolution feature net 61x61 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    inputs = Input(shape=(n_channels, 61, 61))
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=61)(inputs)
    conv1 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(img_norm)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    pool1 = MaxPool2D(pool_size=(2, 2))(act2)

    conv3 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    pool2 = MaxPool2D(pool_size=(2, 2))(act4)

    conv5 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    pool3 = MaxPool2D(pool_size=(2, 2))(act6)

    side_conv0 = Conv2D(64, (59, 59), dilation_rate=1, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(conv1)
    side_norm0 = BatchNormalization(axis=channel_axis)(side_conv0)
    side_act0 = Activation('relu')(side_norm0)

    side_conv1 = Conv2D(64, (28, 28), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
    side_norm1 = BatchNormalization(axis=channel_axis)(side_conv1)
    side_act1 = Activation('relu')(side_norm1)

    side_conv2 = Conv2D(64, (12, 12), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
    side_norm2 = BatchNormalization(axis=channel_axis)(side_conv2)
    side_act2 = Activation('relu')(side_norm2)

    side_conv3 = Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool3)
    side_norm3 = BatchNormalization(axis=channel_axis)(side_conv3)
    side_act3 = Activation('relu')(side_norm3)

    merge_layer1 = Concatenate(axis=channel_axis)([side_act0, side_act1, side_act2, side_act3])

    tensor_prod1 = TensorProd2D(256, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(merge_layer1)
    norm7 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act7 = Activation('relu')(norm7)

    tensor_prod2 = TensorProd2D(256, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)
    flat = Flatten()(tensor_prod2)
    act8 = Activation('softmax')(flat)

    model = Model(inputs=inputs, outputs=act8)

    return model


def dilated_bn_multires_feature_net_61x61(input_shape=(2, 1080, 1280), n_features=3, reg=1e-5, init='he_normal', softmax=False, location=False, weights_path=None, from_logits=False, norm_method='std'):
    print('Using dilated multi-resolution feature net 61x61 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    d = 1
    inputs = Input(shape=input_shape)
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=61)(inputs)
    conv1 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(img_norm)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv2D(64, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    pool1 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act2)
    d *= 2

    conv3 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    pool2 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act4)
    d *= 2

    conv5 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    pool3 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act6)
    d *= 2

    # side_conv0 = Conv2D(64, (59, 59), dilation_rate=1, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(conv1)
    # side_norm0 = BatchNormalization(axis = 1)(side_conv0)
    # side_act0 = Activation('relu')(side_norm0)

    side_conv1 = Conv2D(64, (28, 28), dilation_rate=2, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
    side_norm1 = BatchNormalization(axis=channel_axis)(side_conv1)
    side_act1 = Activation('relu')(side_norm1)

    side_conv2 = Conv2D(64, (12, 12), dilation_rate=4, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
    side_norm2 = BatchNormalization(axis=channel_axis)(side_conv2)
    side_act2 = Activation('relu')(side_norm2)

    side_conv3 = Conv2D(64, (4, 4), dilation_rate=8, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool3)
    side_norm3 = BatchNormalization(axis=channel_axis)(side_conv3)
    side_act3 = Activation('relu')(side_norm3)

    merge_layer1 = Concatenate(axis=channel_axis)([side_act1, side_act2, side_act3])

    tensor_prod1 = TensorProd2D(256, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(merge_layer1)
    norm7 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act7 = Activation('relu')(norm7)

    tensor_prod2 = TensorProd2D(256, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)

    if softmax:
        act8 = Softmax(axis=channel_axis)(tensor_prod2)
    else:
        act8 = tensor_prod2

    final_layer = act8

    model = Model(inputs=inputs, outputs=final_layer)

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def bn_multires_feature_net(input_shape=(2, 1080, 1280), batch_shape=None, n_features=3, reg=1e-5, init='he_normal', softmax=True, location=False, norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if batch_shape is None:
        input1 = Input(shape=input_shape)
    else:
        input1 = Input(batch_shape=batch_shape)
        input_shape = batch_shape[1:]

    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=filter_size)(input1)

    if location:
        loc0 = Location(in_shape=input_shape)(img_norm)
        input2 = Concatenate(axis=channel_axis)([img_norm, loc0])
    else:
        input2 = img_norm

    conv1 = Conv2D(32, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(input2)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv2D(32, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)

    conv3 = Conv2D(32, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act2)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv2D(32, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)

    conv5 = Conv2D(32, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act4)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv2D(32, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)

    conv7 = Conv2D(32, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act6)
    norm7 = BatchNormalization(axis=channel_axis)(conv7)
    act7 = Activation('relu')(norm7)

    conv8 = Conv2D(32, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act7)
    norm8 = BatchNormalization(axis=channel_axis)(conv8)
    act8 = Activation('relu')(norm8)

    conv9 = Conv2D(32, (3, 3), dilation_rate=16, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act8)
    norm9 = BatchNormalization(axis=channel_axis)(conv9)
    act9 = Activation('relu')(norm9)

    conv10 = Conv2D(32, (3, 3), dilation_rate=16, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act9)
    norm10 = BatchNormalization(axis=channel_axis)(conv10)
    act10 = Activation('relu')(norm10)

    conv11 = Conv2D(32, (3, 3), dilation_rate=32, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act10)
    norm11 = BatchNormalization(axis=channel_axis)(conv11)
    act11 = Activation('relu')(norm11)

    conv12 = Conv2D(32, (3, 3), dilation_rate=32, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act11)
    norm12 = BatchNormalization(axis=channel_axis)(conv12)
    act12 = Activation('relu')(norm12)

    merge1 = Concatenate(axis=channel_axis)([act1, act2, act3, act4, act5, act6, act7, act8, act9, act10, act11, act12])

    tensor_prod1 = TensorProd2D(32 * 12, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(merge1)
    norm9 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act9 = Activation('relu')(norm9)

    tensor_prod2 = TensorProd2D(128, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(act9)
    norm10 = BatchNormalization(axis=channel_axis)(tensor_prod2)
    act10 = Activation('relu')(norm10)

    tensor_prod3 = TensorProd2D(128, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act10)

    if softmax:
        act11 = Softmax(axis=channel_axis)(tensor_prod3)
    else:
        act11 = tensor_prod3

    final_layer = act11

    model = Model(inputs=input1, outputs=final_layer)

    return model


def bn_multires_pool_feature_net(input_shape=(2, 1080, 1280), n_features=3, reg=1e-5, init='he_normal', norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    input1 = Input(shape=input_shape)
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=filter_size)(input1)
    conv1 = Conv2D(32, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(img_norm)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv2D(32, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)

    pool1 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(act2)

    conv3 = Conv2D(32, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(pool1)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv2D(32, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)

    pool2 = MaxPool2D(pool_size=(4, 4), strides=1, padding='same')(act4)

    conv5 = Conv2D(32, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(pool2)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv2D(32, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)

    pool3 = MaxPool2D(pool_size=(8, 8), strides=1, padding='same')(act6)

    conv7 = Conv2D(32, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(pool3)
    norm7 = BatchNormalization(axis=channel_axis)(conv7)
    act7 = Activation('relu')(norm7)

    conv8 = Conv2D(32, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act7)
    norm8 = BatchNormalization(axis=channel_axis)(conv8)
    act8 = Activation('relu')(norm8)

    pool4 = MaxPool2D(pool_size=(16, 16), strides=1, padding='same')(act8)

    conv9 = Conv2D(32, (3, 3), dilation_rate=16, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(pool3)
    norm9 = BatchNormalization(axis=channel_axis)(conv9)
    act9 = Activation('relu')(norm9)

    conv10 = Conv2D(32, (3, 3), dilation_rate=16, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act9)
    norm10 = BatchNormalization(axis=channel_axis)(conv10)
    act10 = Activation('relu')(norm10)

    pool5 = MaxPool2D(pool_size=(32, 32), strides=1, padding='same')(act10)

    conv11 = Conv2D(32, (3, 3), dilation_rate=32, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(pool5)
    norm11 = BatchNormalization(axis=channel_axis)(conv11)
    act11 = Activation('relu')(norm11)

    conv12 = Conv2D(32, (3, 3), dilation_rate=32, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act11)
    norm12 = BatchNormalization(axis=channel_axis)(conv12)
    act12 = Activation('relu')(norm12)

    merge1 = Concatenate(axis=channel_axis)([act1, act2, act3, act4, act5, act6, act7, act8, act9, act10, act11, act12, pool1, pool2, pool3, pool4, pool5])

    tensor_prod1 = TensorProd2D(32 * 12, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(merge1)
    norm9 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act9 = Activation('relu')(norm9)

    tensor_prod2 = TensorProd2D(128, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(act9)
    norm10 = BatchNormalization(axis=channel_axis)(tensor_prod2)
    act10 = Activation('relu')(norm10)

    tensor_prod3 = TensorProd2D(128, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act10)
    act11 = Softmax(axis=channel_axis)(tensor_prod3)

    final_layer = act11

    model = Model(inputs=input1, outputs=final_layer)

    return model


def bn_dense_feature_net(input_shape=(2, 1080, 1280), batch_shape=None, n_features=3, reg=1e-5, init='he_normal', softmax=True, location=False, norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if batch_shape is None:
        input1 = Input(shape=input_shape)
    else:
        input1 = Input(batch_shape=batch_shape)
        input_shape = batch_shape[1:]

    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=filter_size)(input1)

    if location is True:
        loc0 = Location(in_shape=input_shape)(img_norm)
        input2 = Concatenate(axis=channel_axis)([img_norm, loc0])
    else:
        input2 = img_norm

    conv1 = Conv2D(48, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(input2)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)
    merge1 = Concatenate(axis=channel_axis)([input2, act1])

    conv2 = Conv2D(48, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    merge2 = Concatenate(axis=channel_axis)([merge1, act2])

    conv3 = Conv2D(48, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge2)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)
    merge3 = Concatenate(axis=channel_axis)([merge2, act3])

    conv4 = Conv2D(48, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    merge4 = Concatenate(axis=channel_axis)([merge3, act4])

    conv5 = Conv2D(48, (3, 3), dilation_rate=16, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge4)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)
    merge5 = Concatenate(axis=channel_axis)([merge4, act5])

    conv6 = Conv2D(48, (3, 3), dilation_rate=32, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    merge6 = Concatenate(axis=channel_axis)([merge5, act6])

    tensor_prod1 = TensorProd2D(48 * 6 + input_shape[0], 256, kernel_initializer=init, kernel_regularizer=l2(reg))(merge6)
    norm9 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act9 = Activation('relu')(norm9)

    tensor_prod2 = TensorProd2D(256, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(act9)
    norm10 = BatchNormalization(axis=channel_axis)(tensor_prod2)
    act10 = Activation('relu')(norm10)

    tensor_prod3 = TensorProd2D(256, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act10)

    if softmax:
        tensor_prod3 = Softmax(axis=channel_axis)(tensor_prod3)

    final_layer = tensor_prod3

    model = Model(inputs=input1, outputs=final_layer)

    return model


def disc_net(input_shape=(256, 256, 1), seg_model=None, n_features=16, reg=1e-5, init='he_normal', softmax=True, location=True, norm_method='std', filter_size=61):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    for layer in seg_model.layers:
        layer.trainable = False

    input1 = Input(shape=input_shape)
    seg_output = seg_model(input1)
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=filter_size)(input1)

    if location:
        loc0 = Location(in_shape=input_shape)(img_norm)
        input2 = Concatenate(axis=channel_axis)([img_norm, loc0])
    else:
        input2 = img_norm

    input3 = Concatenate(axis=channel_axis)([input2, seg_output])

    conv1 = Conv2D(32, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(input3)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv2D(32, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)

    conv3 = Conv2D(32, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act2)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv2D(32, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)

    conv5 = Conv2D(32, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act4)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv2D(32, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)

    conv7 = Conv2D(32, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act6)
    norm7 = BatchNormalization(axis=channel_axis)(conv7)
    act7 = Activation('relu')(norm7)

    conv8 = Conv2D(32, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act7)
    norm8 = BatchNormalization(axis=channel_axis)(conv8)
    act8 = Activation('relu')(norm8)

    conv9 = Conv2D(32, (3, 3), dilation_rate=16, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act8)
    norm9 = BatchNormalization(axis=channel_axis)(conv9)
    act9 = Activation('relu')(norm9)

    conv10 = Conv2D(32, (3, 3), dilation_rate=16, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act9)
    norm10 = BatchNormalization(axis=channel_axis)(conv10)
    act10 = Activation('relu')(norm10)

    conv11 = Conv2D(32, (3, 3), dilation_rate=32, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act10)
    norm11 = BatchNormalization(axis=channel_axis)(conv11)
    act11 = Activation('relu')(norm11)

    conv12 = Conv2D(32, (3, 3), dilation_rate=32, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act11)
    norm12 = BatchNormalization(axis=channel_axis)(conv12)
    act12 = Activation('relu')(norm12)

    conv13 = Conv2D(32, (3, 3), dilation_rate=64, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act12)
    norm13 = BatchNormalization(axis=channel_axis)(conv13)
    act13 = Activation('relu')(norm13)

    conv14 = Conv2D(32, (3, 3), dilation_rate=64, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act13)
    norm14 = BatchNormalization(axis=channel_axis)(conv14)
    act14 = Activation('relu')(norm14)

    merge1 = Concatenate(axis=channel_axis)([act1, act2, act3, act4, act5, act6, act7, act8, act9, act10, act11, act12, act13, act14])

    tensor_prod1 = TensorProd2D(32 * 12, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(merge1)
    norm9 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act9 = Activation('relu')(norm9)

    tensor_prod2 = TensorProd2D(128, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(act9)
    norm10 = BatchNormalization(axis=channel_axis)(tensor_prod2)
    act10 = Activation('relu')(norm10)

    tensor_prod3 = TensorProd2D(128, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act10)

    if softmax:
        act15 = Softmax(axis=channel_axis)(tensor_prod3)
    else:
        act15 = tensor_prod3

    final_layer = act15

    model = Model(inputs=input1, outputs=final_layer)

    return model


"""
Residual layers for fine tuning
"""


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def dilated_identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=1):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        dilation_rate: dilation rate
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', dilation_rate=dilation_rate, name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def ASPP_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x_in = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', dilation_rate=1, name=conv_name_base + '2b')(x_in)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x_1 = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', dilation_rate=2, name=conv_name_base + '2c')(x_in)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x_2 = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', dilation_rate=4, name=conv_name_base + '2d')(x_in)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2d')(x)
    x_3 = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', dilation_rate=8, name=conv_name_base + '2e')(x_in)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2e')(x)
    x_4 = Activation('relu')(x)

    x = Concatenate(axis=bn_axis)([x_1, x_2, x_3, x_4])

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2f')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2f')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def resnet_custom(input_shape=(2, 512, 512), batch_shape=None, n_features=3, reg=1e-5, init='he_normal', upsample=True, softmax=False, norm_method='std', filter_size=61):
    print("Using resnet_custom")

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if batch_shape is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = Input(batch_shape=batch_shape)
        input_shape = batch_shape[1:]
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=filter_size)(inputs)
    loc0 = Location(in_shape=input_shape)(img_norm)
    merge0 = Concatenate(axis=bn_axis)([img_norm, loc0])

    # inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(merge0)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = ASPP_block(x, 3, [256, 256, 512], stage=4, block='a')

    # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # x = AvgPool2D((7, 7), name='avg_pool')(x)
    x = Resize(scale=8)(x)
    x = TensorProd2D(1024, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(x)

    if softmax:
        x = Softmax(axis=bn_axis)(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def dilated_bn_res_feature_net_61x61(input_shape=(2, 1080, 1280), n_features=3, reg=1e-5, init='he_normal', weights_path=None, from_logits=False, norm_method='std'):
    print("Using dilated multi-resolution feature net 61x61 with batch normalization")

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    d = 1
    inputs = Input(shape=input_shape)
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=61)(inputs)
    conv1 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(img_norm)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv2D(64, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    pool1 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act2)
    d *= 2

    conv3 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    pool2 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act4)
    d *= 2

    conv5 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    pool3 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act6)
    d *= 2

    side_conv3 = Conv2D(64, (4, 4), dilation_rate=8, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool3)
    side_norm3 = BatchNormalization(axis=channel_axis)(side_conv3)
    side_act3 = Activation('relu')(side_norm3)

    tensor_prod1 = TensorProd2D(64, 64, kernel_initializer=init, kernel_regularizer=l2(reg))(side_act3)
    norm7 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act7 = Activation('relu')(norm7)

    conv_red = Conv2D(64, (1, 1), strides=(2, 2), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act7)
    x = Resize(scale=2)(conv_red)
    # x = ASPP_block(act7, 3, [64, 64, 64], stage = 1, block = 'a')
    # x = ASPP_block(x, 3, [64, 64, 64], stage = 2, block = 'b')

    tensor_prod2 = TensorProd2D(64, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(x)

    act8 = Softmax(axis=channel_axis)(tensor_prod2)

    final_layer = act8

    model = Model(inputs=inputs, outputs=final_layer)

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


"""
Multiple input conv-nets for fully convolutional training
"""


def dilated_bn_feature_net_gather_61x61(input_shape=(2, 1080, 1280), training_examples=1e5, batch_size=None, n_features=3, reg=1e-5, init='he_normal', weights_path=None, norm_method='std'):
    print('Using dilated feature net 61x61 with batch normalization')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    input1 = Input(shape=input_shape)

    d = 1
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=61)(input1)
    conv1 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', batch_size=batch_size, kernel_regularizer=l2(reg))(img_norm)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv2D(64, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    pool1 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act2)
    d *= 2

    conv3 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    pool2 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act4)
    d *= 2

    conv5 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    pool3 = DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(act6)
    d *= 2

    conv7 = Conv2D(200, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool3)
    norm7 = BatchNormalization(axis=channel_axis)(conv7)
    act7 = Activation('relu')(norm7)

    tensorprod1 = TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)
    norm8 = BatchNormalization(axis=channel_axis)(tensorprod1)
    act8 = Activation('relu')(norm8)

    tensorprod2 = TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act8)
    act9 = Softmax(axis=channel_axis)(tensorprod2)

    batch_index_input = Input(batch_shape=(training_examples,), dtype='int32')
    row_index_input = Input(batch_shape=(training_examples,), dtype='int32')
    col_index_input = Input(batch_shape=(training_examples,), dtype='int32')

    index1 = K.stack([batch_index_input, row_index_input, col_index_input], axis=1)

    def gather_indices(x):
        return tf.gather_nd(x, index1)

    gather1 = Lambda(gather_indices)(act9)

    model = Model(inputs=[input1, batch_index_input, row_index_input, col_index_input], outputs=[gather1])

    print(model.output_shape)
    return model


"""
3D Conv-nets
"""


def multires_block(input_tensor, num_filters=16, init='he_normal', reg=1e-5):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    conv1 = Conv3D(num_filters, (1, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(input_tensor)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)
    merge1 = Concatenate(axis=channel_axis)([input_tensor, act1])

    conv2 = Conv3D(num_filters, (1, 3, 3), dilation_rate=(1, 2, 2), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    merge2 = Concatenate(axis=channel_axis)([merge1, act2])

    conv3 = Conv3D(num_filters, (1, 3, 3), dilation_rate=(1, 4, 4), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge2)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)
    merge3 = Concatenate(axis=channel_axis)([merge2, act3])

    conv4 = Conv3D(num_filters, (1, 3, 3), dilation_rate=(1, 8, 8), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    merge4 = Concatenate(axis=channel_axis)([merge3, act4])

    conv5 = Conv3D(num_filters, (1, 3, 3), dilation_rate=(1, 16, 16), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge4)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)
    merge5 = Concatenate(axis=channel_axis)([merge4, act5])

    conv6 = Conv3D(num_filters, (1, 3, 3), dilation_rate=(1, 32, 32), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    merge6 = Concatenate(axis=channel_axis)([merge5, act6])

    return merge6


def bn_dense_multires_feature_net_3D(batch_shape=(1, 1, 10, 256, 256), n_blocks=10, n_features=3, reg=1e-5, init='he_normal', norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    input1 = Input(batch_shape=batch_shape)
    img_norm = ImageNormalization3D(norm_method=norm_method, filter_size=filter_size)(input1)
    list_of_blocks = []
    list_of_blocks.append(multires_block(img_norm, init=init, reg=reg))

    for _ in range(n_blocks - 1):
        list_of_blocks.append(multires_block(list_of_blocks[-1], init=init, reg=reg))

    tensor_prod1 = TensorProd3D(n_blocks * 6 + batch_shape[1], 64, kernel_initializer=init, kernel_regularizer=l2(reg))(list_of_blocks[-1])
    norm1 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act1 = Activation('relu')(norm1)

    tensor_prod2 = TensorProd3D(64, 64, kernel_initializer=init, kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(tensor_prod2)
    act2 = Activation('relu')(norm2)

    tensor_prod3 = TensorProd3D(64, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act2)

    if softmax:
        tensor_prod3 = Softmax(axis=channel_axis)(tensor_prod3)

    final_layer = tensor_prod3

    model = Model(inputs=input1, outputs=final_layer)

    return model


def bn_feature_net_3D(batch_shape=(1, 1, 10, 256, 256), n_features=3, reg=1e-5, init='he_normal', location=False, softmax=True, norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    input1 = Input(batch_shape=batch_shape)
    input_shape = batch_shape[1:]

    img_norm = ImageNormalization3D(norm_method=norm_method, filter_size=filter_size)(input1)

    if location:
        loc0 = Location3D(in_shape=batch_shape)(img_norm)
        input2 = Concatenate(axis=channel_axis)([img_norm, loc0])
    else:
        input2 = img_norm

    conv1 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(input2)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)

    conv2 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 2, 2), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)

    conv3 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 4, 4), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act2)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)

    conv4 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 8, 8), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)

    conv5 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 16, 16), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act4)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)

    conv6 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 32, 32), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)

    merge1 = Concatenate(axis=channel_axis)([act1, act2, act3, act4, act5, act6])

    tensor_prod1 = TensorProd3D(64 * 6, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(merge1)
    norm7 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act7 = Activation('relu')(norm7)

    tensor_prod2 = TensorProd3D(256, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)
    norm8 = BatchNormalization(axis=channel_axis)(tensor_prod2)
    act8 = Activation('relu')(norm8)

    tensor_prod3 = TensorProd3D(256, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act8)

    if softmax:
        tensor_prod3 = Softmax(axis=channel_axis)(tensor_prod3)

    final_layer = tensor_prod3

    model = Model(inputs=input1, outputs=final_layer)

    return model


def bn_dense_feature_net_3D(batch_shape=(1, 1, 5, 256, 256), n_features=3, reg=1e-5, init='he_normal', location=False, softmax=True, norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    input1 = Input(batch_shape=batch_shape)
    input_shape = batch_shape[1:]

    img_norm = ImageNormalization3D(norm_method=norm_method, filter_size=filter_size)(input1)

    if location:
        loc0 = Location3D(in_shape=batch_shape)(img_norm)
        input2 = Concatenate(axis=channel_axis)([img_norm, loc0])
    else:
        input2 = img_norm

    conv1 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(input2)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)
    merge1 = Concatenate(axis=channel_axis)([input2, act1])

    conv2 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 2, 2), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    merge2 = Concatenate(axis=channel_axis)([merge1, act2])

    conv3 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 4, 4), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge2)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)
    merge3 = Concatenate(axis=channel_axis)([merge2, act3])

    conv4 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 8, 8), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    merge4 = Concatenate(axis=channel_axis)([merge3, act4])

    conv5 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 16, 16), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge4)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)
    merge5 = Concatenate(axis=channel_axis)([merge4, act5])

    conv6 = Conv3D(64, (5, 3, 3), dilation_rate=(1, 32, 32), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    merge6 = Concatenate(axis=channel_axis)([merge5, act6])

    tensor_prod1 = TensorProd3D(64 * 6 + input_shape[0], 256, kernel_initializer=init, kernel_regularizer=l2(reg))(merge6)
    norm7 = BatchNormalization(axis=channel_axis)(tensor_prod1)
    act7 = Activation('relu')(norm7)

    tensor_prod2 = TensorProd3D(256, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)
    norm8 = BatchNormalization(axis=channel_axis)(tensor_prod2)
    act8 = Activation('relu')(norm8)

    tensor_prod3 = TensorProd3D(256, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act8)

    if softmax:
        tensor_prod3 = Softmax(axis=channel_axis)(tensor_prod3)

    final_layer = tensor_prod3

    model = Model(inputs=input1, outputs=final_layer)

    return model


def bn_dense_feature_net_lstm(input_shape=(1, 60, 256, 256), batch_shape=None, n_features=3, reg=1e-5, init='he_normal', softmax=True, norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if batch_shape is None:
        input1 = Input(shape=input_shape)
    else:
        input1 = Input(batch_shape=batch_shape)
        input_shape = batch_shape[1:]

    img_norm = ImageNormalization3D(norm_method=norm_method, filter_size=filter_size)(input1)

    conv1 = Conv3D(64, (1, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(img_norm)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    act1 = Activation('relu')(norm1)
    merge1 = Concatenate(axis=channel_axis)([img_norm, act1])

    conv2 = Conv3D(64, (1, 3, 3), dilation_rate=(1, 2, 2), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge1)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    act2 = Activation('relu')(norm2)
    merge2 = Concatenate(axis=channel_axis)([merge1, act2])

    conv3 = Conv3D(64, (1, 3, 3), dilation_rate=(1, 4, 4), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge2)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    act3 = Activation('relu')(norm3)
    merge3 = Concatenate(axis=channel_axis)([merge2, act3])

    conv4 = Conv3D(64, (1, 3, 3), dilation_rate=(1, 8, 8), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge3)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    act4 = Activation('relu')(norm4)
    merge4 = Concatenate(axis=channel_axis)([merge3, act4])

    conv5 = Conv3D(64, (1, 3, 3), dilation_rate=(1, 16, 16), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge4)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)
    act5 = Activation('relu')(norm5)
    merge5 = Concatenate(axis=channel_axis)([merge4, act5])

    conv6 = Conv3D(64, (1, 3, 3), dilation_rate=(1, 32, 32), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(merge5)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)
    act6 = Activation('relu')(norm6)
    merge6 = Concatenate(axis=channel_axis)([merge5, act6])

    tensorprod1 = TensorProd2D(64 * 6, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(merge6)

    lstm1 = ConvLSTM2D(64, (3, 3), dilation_rate=(1, 1), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True)(tensorprod1)
    lstm2 = ConvLSTM2D(64, (3, 3), dilation_rate=(1, 1), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True, go_backwards=True)(lstm1)

    lstm3 = ConvLSTM2D(64, (3, 3), dilation_rate=(2, 2), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True)(lstm2)
    lstm4 = ConvLSTM2D(64, (3, 3), dilation_rate=(2, 2), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True, go_backwards=True)(lstm3)

    lstm5 = ConvLSTM2D(64, (3, 3), dilation_rate=(4, 4), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True)(lstm4)
    lstm6 = ConvLSTM2D(64, (3, 3), dilation_rate=(4, 4), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True, go_backwards=True)(lstm5)

    lstm7 = ConvLSTM2D(64, (3, 3), dilation_rate=(8, 8), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True)(lstm6)
    lstm8 = ConvLSTM2D(64, (3, 3), dilation_rate=(8, 8), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True, go_backwards=True)(lstm7)

    lstm9 = ConvLSTM2D(64, (3, 3), dilation_rate=(16, 16), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True)(lstm8)
    lstm10 = ConvLSTM2D(64, (3, 3), dilation_rate=(16, 16), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True, go_backwards=True)(lstm9)

    lstm11 = ConvLSTM2D(64, (3, 3), dilation_rate=(32, 32), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True)(lstm10)
    lstm12 = ConvLSTM2D(64, (3, 3), dilation_rate=(32, 32), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), return_sequences=True, go_backwards=True)(lstm11)

    final_layer = lstm12

    model = Model(inputs=input1, outputs=final_layer)

    return model


def siamese_model(input_shape=None, track_length=1, batch_shape=None, reg=1e-5, init='he_normal', softmax=True, norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        new_input_shape = tuple([input_shape[0]] + [None] + list(input_shape[1:]))
    else:
        channel_axis = -1
        new_input_shape = tuple([None] + list(input_shape))
        
    input_shape = new_input_shape

    # Define the input shape for the images
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)
    # Define the input shape for the other data (centroids, etc)
    input_3 = Input(shape=(None, 2))
    input_4 = Input(shape=(None, 2))

    # Sequential interface for siamese portion of model
    feature_extractor = Sequential()
    feature_extractor.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg), input_shape=input_shape))
    feature_extractor.add(BatchNormalization(axis=channel_axis))
    feature_extractor.add(Activation('relu'))
    feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))
    
    feature_extractor.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    feature_extractor.add(BatchNormalization(axis=channel_axis))
    feature_extractor.add(Activation('relu'))
    feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))
    
    feature_extractor.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    feature_extractor.add(BatchNormalization(axis=channel_axis))
    feature_extractor.add(Activation('relu'))
    feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))

    feature_extractor.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    feature_extractor.add(BatchNormalization(axis=channel_axis))
    feature_extractor.add(Activation('relu'))
    feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))
    
    feature_extractor.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    feature_extractor.add(BatchNormalization(axis=channel_axis))
    feature_extractor.add(Activation('relu'))
    feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))
    
    feature_extractor.add(Reshape(tuple([-1, 64])))

    # Create two instances of feature_extractor
    output_1 = feature_extractor(input_1)
    output_2 = feature_extractor(input_2)
    
    lstm_1 = LSTM(64)(output_1)
    output_2_reshape = Reshape((64,))(output_2)
    
    lstm_3 = LSTM(64)(input_3)
    input_4_reshape = Reshape((2,))(input_4)

    # Combine the extracted features with other known features (centroids)
    merge_1 = Concatenate(axis=channel_axis)([lstm_1, output_2_reshape])
    merge_2 = Concatenate(axis=channel_axis)([lstm_3, input_4_reshape])
    
    dense_merge_1 = Dense(128)(merge_1)
    bn_merge_1 = BatchNormalization(axis=channel_axis)(dense_merge_1)
    dense_relu_1 = Activation('relu')(bn_merge_1)
    
    dense_merge_2 = Dense(128)(merge_2)
    bn_merge_2 = BatchNormalization(axis=channel_axis)(dense_merge_2)
    dense_relu_2 = Activation('relu')(bn_merge_2)
    
    # Concatenate outputs from both instances
    merged_outputs = Concatenate(axis=channel_axis)([dense_relu_1, dense_relu_2])

    # Implement dense net (Alternatively, could call preexisting) with the 2 merged outputs as inputs
    dense1 = Dense(128)(merged_outputs)
    bn1 = BatchNormalization(axis=channel_axis)(dense1)
    relu1 = Activation('relu')(bn1)
    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization(axis=channel_axis)(dense2)
    relu2 = Activation('relu')(bn2)
    dense3 = Dense(3, activation='softmax')(relu2)

    # Instantiate model
    final_layer = dense3
    model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=final_layer)
    
    return model


def watershed_net(input_shape=(256, 256, 1), n_features=16, reg=1e-5, init='he_normal', norm_method='std', filter_size=61):
    inputs1 = Input(input_shape)
    img_norm = ImageNormalization2D(norm_method=norm_method, filter_size=filter_size, input_shape=input_shape)(inputs1)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=init)(img_norm)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=init)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=init)(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=init)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=init)(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=init)(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=init)(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=init)(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=init)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=init)(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=init)(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=init)(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=init)(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=init)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=init)(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=init)(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=init)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs1, outputs=conv10)
    return model


def bn_feature_net_31x31_3D(n_features=3, n_frames=5, n_channels=1, reg=1e-5, init='he_normal', norm_method='whole_image'):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        input_shape = (n_channels, n_frames, 31, 31)
    else:
        channel_axis = -1
        input_shape = (n_frames, 31, 31, n_channels)

    model = Sequential()

    model.add(ImageNormalization3D(norm_method=norm_method, filter_size=31, input_shape=input_shape))
    model.add(Conv3D(64, (1, 4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg), input_shape=input_shape))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(200, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(200, (n_frames, 1, 1), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Flatten())

    model.add(Activation('softmax'))

    return model


def dilated_bn_feature_net_31x31_3D(input_shape=(2, 1080, 1280), n_frames=5, n_channels=None, flatten=False, batch_size=None, n_features=3, reg=1e-5, init='he_normal', weights_path=None, norm_method='whole_image'):

    if n_channels is not None:
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
            input_shape = (n_channels, n_frames, 31, 31)
        else:
            channel_axis = -1
            input_shape = (n_frames, 31, 31, n_channels)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    model = Sequential()

    d = 1
    model.add(ImageNormalization3D(norm_method=norm_method, filter_size=31, input_shape=input_shape))
    model.add(Conv3D(64, (1, 4, 4), input_shape=input_shape, dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool3D(dilation_rate=(1, d, d), pool_size=(1, 2, 2)))
    d *= 2

    model.add(Conv3D(64, (1, 3, 3), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 3, 3), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool3D(dilation_rate=(1, d, d), pool_size=(1, 2, 2)))
    d *= 2

    model.add(Conv3D(64, (1, 3, 3), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(200, (1, 3, 3), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(200, (n_frames, 1, 1), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

    if flatten:
        model.add(Flatten())

    model.add(Activation('softmax'))

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def bn_feature_net_61x61_3D(n_features=3, n_channels=1, n_frames=1, reg=1e-5, init='he_normal', norm_method='whole_image'):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        input_shape = (n_channels, n_frames, 61, 61)
    else:
        channel_axis = -1
        input_shape = (n_frames, 61, 61, n_channels)

    model = Sequential()

    model.add(ImageNormalization3D(norm_method=norm_method, filter_size=61, input_shape=input_shape))
    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(200, (1, 4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(200, (n_frames, 1, 1), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Flatten())

    model.add(Softmax(axis=channel_axis))

    return model


def dilated_bn_feature_net_61x61_3D(input_shape=(1, 1080, 1280, 2), n_frames=1, n_channels=None, flatten=False, batch_size=None, n_features=3, reg=1e-5, init='he_normal', weights_path=None, norm_method='whole_image'):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    model = Sequential()

    d = 1
    model.add(ImageNormalization3D(norm_method=norm_method, filter_size=61, input_shape=input_shape))
    model.add(Conv3D(64, (1, 3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool3D(dilation_rate=(1, d, d), pool_size=(1, 2, 2)))
    d *= 2

    model.add(Conv3D(64, (1, 3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool3D(dilation_rate=(1, d, d), pool_size=(1, 2, 2)))
    d *= 2

    model.add(Conv3D(64, (1, 3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1, 3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(DilatedMaxPool3D(dilation_rate=(1, d, d), pool_size=(1, 2, 2)))
    d *= 2

    model.add(Conv3D(200, (1, 4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv3D(200, (n_frames, 1, 1), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProd3D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

    if flatten:
        model.add(Flatten())

    model.add(Softmax(axis=channel_axis))

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model
