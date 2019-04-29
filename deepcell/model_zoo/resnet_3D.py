# 3D adaptation of ResNet50, based on the 2D version of : https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

from scipy import interpolate
from skimage import io
from keras.layers import Input, Add, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D, AveragePooling3D, MaxPooling3D
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import numpy as np
import deepcell
import keras.backend as K
K.set_learning_phase(1)

def interpolation_z_axis(data, multiplicator):
    result_img = np.empty((data.shape[0], data.shape[1] * multiplicator, data.shape[2], data.shape[3]))

    for i in range(data.shape[0]):
        print(i)
        X = data[i]
        temp_image = np.empty((X.shape[0] * multiplicator, X.shape[1], X.shape[2]))
        temp_image[:] = np.nan
        for j in range(X.shape[0]):
            temp_image[j * multiplicator] = X[j]
        temp_image[-1] = temp_image[-2]
        indexes = np.arange(temp_image.shape[0])
        good = np.isfinite(temp_image).all(axis=(1, 2))
        f = interpolate.interp1d(indexes[good], temp_image[good], bounds_error=False, axis=0)
        B = f(indexes)
        result_img[i] = B

    io.imsave('InterpolatedX_train_' + str(multiplicator) + '.tiff', result_img.astype('uint8'), 'tifffile')
    print("Final shape : ", result_img.shape)
    return result_img.astype('uint8')




def identity_block(X, f, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value.
    X_shortcut = X

    # First component of main path
    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv3D(F1, (1, 1, 1), strides=(s, s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X_shortcut)
    X_shortcut = BatchNormalization(axis=1, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape, classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    print("Input shape : ", X_input.shape)

    # Zero-Padding
    X = ZeroPadding3D(padding=(3, 3, 3), data_format = 'channels_first')(X_input)
    print("X shape Zero Padding : ", X.shape)

    # Stage 1
    X = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0), data_format = 'channels_first')(X)
    print("X shape conv3D: ", X.shape)

    X = BatchNormalization(axis=1, name='bn_conv1')(X)
    print("X shape BatchNormalization: ", X.shape)

    X = Activation('relu')(X)
    print("X shape Activation: ", X.shape)

    X = MaxPooling3D((3, 3, 3), strides=(1, 1, 1))(X)
    print("X shape Stage 1 : ", X.shape)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 64], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 64], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 64], stage=2, block='c')

    print("X shape Stage 2 : ", X.shape)

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 128], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 128], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 128], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 128], stage=3, block='d')

    print("X shape Stage 3 : ", X.shape)

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 256], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='f')

    print("X shape Stage 4 : ", X.shape)

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 512], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 512], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 512], stage=5, block='c')

    print("X shape Stage 5 : ", X.shape)

    # AVGPOOL
    X = AveragePooling3D((7, 7, 7), name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

def main():

    # Download the data (saves to ~/.keras/datasets)
    filename = 'mousebrain.npz'
    (X_train, y_train), (X_test, y_test) = deepcell.datasets.mousebrain.load_data(filename)
    X_train_2 =  X_train[:,:,:,:,0]
    y_train_2 = y_train[:,:,:,:,0]

    print('X_train_2.shape: {}\ny.shape: {}'.format(X_train_2.shape, y_train.shape))

    io.imsave('InterpolatedX_train_' + str('zut') + '.tiff', y_train_2.astype('uint8'), 'tifffile')



    X = X_train_2[1:]
    print("X_train_2[1:]", X.shape)

    X = X[np.newaxis]
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y_train.shape))
    X = X[1:]

    print("X.shape[1:]", X.shape[1:])
    #model = ResNet50(input_shape=X.shape[1:], classes=6
    model = ResNet50(input_shape=(1,110,256,256), classes=6)
    #print(model.summary())

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # X_train_interpol = interpolation_z_axis(X_train[:,:,:,:,0], 8)
    # y_train_interpol = interpolation_z_axis(y_train[:,:,:,:,0] ,8)
    #
    # X_train_interpol = X_train_interpol[:,:,:,:,np.newaxis]
    # y_train_interpol = y_train_interpol[:, :, :, :, np.newaxis]
    # print('X train inerpol : "', X_train_interpol.shape)
    #
    # model.fit(X_train_interpol, y_train_interpol, epochs=25, batch_size=32)
    #
    # preds = model.evaluate(X_test, y_test)
    # print ("Loss = " + str(preds[0]))
    # print ("Test Accuracy = " + str(preds[1]))
    #
    # model.summary()
    #
    # plot_model(model, to_file='model.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))

main()
