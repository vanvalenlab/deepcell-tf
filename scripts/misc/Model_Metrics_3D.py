import os
import numpy as np

import matplotlib.pyplot as plt
from skimage.io import imshow

from skimage.measure import label

import deepcell.datasets
from deepcell import metrics
from tensorflow.keras.optimizers import SGD

# Download the data (saves to ~/.keras/datasets)
#filename = 'mousebrain.npz'
#(X_train, y_train), (X_test, y_test) = deepcell.datasets.mousebrain.load_data(filename)

#print('X.shape: {}\ny.shape: {}'.format(X_train.shape, y_train.shape))

from deepcell import model_zoo

# fgbg_model = model_zoo.bn_feature_net_skip_3D(
#     receptive_field=61,
#     n_features=2,  # segmentation mask (is_cell, is_not_cell)
#     n_frames=3,
#     n_skips=3,
#     n_conv_filters=32,
#     n_dense_filters=128,
#     input_shape=tuple([3] + list(X_train.shape[2:])),
#     multires=False,
#     last_only=False,
#     norm_method='whole_image')

#run_fgbg_model = model_zoo.bn_feature_net_skip_3D(
    # receptive_field=61,
    # n_features=2,
    # n_frames=15,
    # n_skips=3,
    # n_conv_filters=32,
    # n_dense_filters=128,
    # input_shape=tuple(X_test.shape[1:]),
    # multires=False,
    # last_only=False,
    # norm_method='whole_image')

# print(fgbg_model.input_shape)
# #print(run_fgbg_model.input_shape)
#
# fgbg_model.compile(SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),metrics=['accuracy'])
# fgbg_model.load_weights('/home/conv_fgbg_3d_model.h5')

#
# print(X_test.shape)
# X_test, y_test = X_test[:4,:3], y_test[:4,:3]
# print(X_test.shape)

with open("predict_3D_metrics.txt", "r") as file:
    predict = eval(file.readline())

print(predict.shape)

#predict = fgbg_model.predict(X_test)


index = np.random.randint(low=0, high=predict.shape[0])
frame = np.random.randint(low=0, high=predict.shape[1])

fig, axes = plt.subplots(ncols=2, figsize=(15, 15), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(predict[2, 2, ..., 0])
ax[0].set_title('Source Image')

ax[1].imshow(predict[2, 1, ..., 0])
ax[1].set_title('FGBG Prediction')
