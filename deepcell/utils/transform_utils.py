"""
transform_utils.py

Functions for data transformations

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import ndimage as ndi
import ndi.measurements

def rotate_array_0(arr):
    return arr

def rotate_array_90(arr):
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim-1, arr.ndim-2]
    slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)

def rotate_array_180(arr):
    slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]

def rotate_array_270(arr):
    axes_order = list(range(arr.ndim-2)) + [arr.ndim-1, arr.ndim-2]
    slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([
        [1, 0, o_x],
        [0, 1, o_y],
        [0, 0, 1]
    ])
    reset_matrix = np.array([
        [1, 0, -o_x],
        [0, 1, -o_y],
        [0, 0, 1]
    ])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def distance_transform_3d(maskstack,bins=16):
    #Input: (no of frames,x,y) : The stack of masks
    #Output:(no of frames,x,y) : 3D Euclidiean Distance Transform
    def weightmask(maskimg):
        img=label(maskimg)
        img=img.flatten()
        unique, counts = np.unique(img, return_counts=True)
        counts=1/np.sqrt(counts)
        dic=dict(zip(unique, counts))
        dic[0]=0
        img_out=map(dic.get, img)
        img_out=list(img_out)
        return np.reshape(img_out,(maskimg.shape[0],maskimg.shape[1],maskimg.shape[2]))

    ms=weightmask(maskstack)
    epsilon=1e-25
    dt_2dslice=[]
    for mask in ms:
        dt_2dslice.append(ndi.distance_transform_edt(mask))
    dt_2dslice=np.array(dt_2dslice)
    print(dt_2dslice.shape)
    distance=np.zeros(list(ms.shape))
    for k in range(ms.shape[0]):
        adder=[np.square(x-k) for x in range(len(dt_2dslice))]
        for i in range(ms.shape[1]):
            for j in range(ms.shape[2]):
                slicearr=np.square(dt_2dslice[:,i,j])
                temparr=slicearr+adder
                zans=np.argmin(temparr)
                distance[k][i][j]=np.sqrt(np.square(dt_2dslice[zans,i,j])+np.square(zans-k))

    min_dist = np.amin(distance.flatten())
    max_dist = np.amax(distance.flatten())
    bins     = np.linspace(min_dist - epsilon, max_dist + epsilon, num=16)
    distance = np.digitize(distance, bins)

    return distance