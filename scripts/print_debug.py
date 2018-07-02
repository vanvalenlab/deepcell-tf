from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras import backend as K

def data_generator(X, batch, feature_dict=None, mode='sample',
                   labels=None, pixel_x=None, pixel_y=None, win_x=30, win_y=30):
    img_list = []
    l_list = []
    for b in batch:
        img_list.append(X[b])
        l_list.append(labels[b])
    img_list = np.stack(tuple(img_list), axis=0).astype(K.floatx())
    l_list = np.stack(tuple(l_list), axis=0)
    return img_list, l_list

print("Setting Eager mode...")
tfe.enable_eager_execution()
training_data = np.load('/data/npz_data/cells/ecoli/kc_polaris/ecoli_kc_polaris_channels_last_disc.npz')
y = training_data['y']
print(y.shape)
print(np.unique(y).size)

X = training_data['X']
class_weights = training_data['class_weights']
win_x = training_data['win_x']
win_y = training_data['win_y']

total_batch_size = X.shape[0]
num_test = np.int32(np.ceil(np.float(total_batch_size) / 10))
num_train = np.int32(total_batch_size - num_test)
full_batch_size = np.int32(num_test + num_train)

print('Batch Size: {}\nNum Test: {}\nNum Train: {}'.format(total_batch_size, num_test, num_train))

# Split data set into training data and validation data
arr = np.arange(total_batch_size)
arr_shuff = np.random.permutation(arr)

train_ind = arr_shuff[0:num_train]
test_ind = arr_shuff[num_train:]

X_train, y_train = data_generator(X, train_ind, labels=y)
X_test, y_test = data_generator(X, test_ind, labels=y)

        # y_test = np.moveaxis(y_test, 1, 3)
train_dict = {'X': X_train,'y': y_train,'class_weights': class_weights, 'win_x': win_x,'win_y': win_y}
print(y_train.shape)

print(y_train[0,0,0,:])
print(y_train[0,10,10,:])
print(y_train[0,20,20,:])
print(y_train[0,30,30,:])
print(y_train[0,40,40,:])
print(y_train[0,50,50,:])
print(y_train[0,60,60,:])
print(y_train[0,70,70,:])

print(np.sum(y_train[0,:,:,0]))

print(np.unique(y_train))
