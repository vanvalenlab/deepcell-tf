import statistics
import numpy as np
import os
import glob

from skimage.measure import regionprops
from sklearn.model_selection import train_test_split

import deepcell

from deepcell_toolbox.utils import tile_image_3D, untile_image_3D
from deepcell_toolbox.deep_watershed import deep_watershed_3D

from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects

def get_3D_cell_statistics(images, precision=2, channel=0):
    
    if images.ndim != 5:
        raise ValueError('Input images must have ndims = 5 (batch, z, x, y, channels)'
                         'Given input had ndims = {}, with shape '.format(images.ndim, images.shape))
    
    z_dims = []
    x_dims = []
    y_dims = []
    
    for batch in range(images.shape[0]):
        regions = regionprops(images[batch, ..., channel])
        
        for region in regions:
            bbox = region.bbox
            z_dims.append(bbox[3] - bbox[0])
            x_dims.append(bbox[4] - bbox[1])
            y_dims.append(bbox[5] - bbox[2])

    stats = {}
    stats['z_mean'] = np.round(statistics.mean(z_dims), precision)
    stats['x_mean'] = np.round(statistics.mean(x_dims), precision)
    stats['y_mean'] = np.round(statistics.mean(y_dims), precision)
    stats['z_med'] = np.round(statistics.median(z_dims), precision)
    stats['x_med'] = np.round(statistics.median(x_dims), precision)
    stats['y_med'] = np.round(statistics.median(y_dims), precision)     
    return stats


def load_mousebrain_data(filepath, set_nums, test_size=0.2, seed=0):
    X = []
    y = []

    for set_num in range(set_nums):
        path_to_folder = os.path.join(filepath, 'mov_{}'.format(set_num))

        for path_to_npz in glob.glob(os.path.join(path_to_folder, '*.npz')):
            with np.load(path_to_npz) as load_data:
                X.append(np.expand_dims(load_data['X'][..., 0], axis=-1))
                y.append(load_data['y'])

    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test


def tile_predict_watershed(model,
                           X_test,
                           y_true,
                           input_shape,
                           stride_ratio=0.5,
                           min_distance=10,
                           detection_threshold=0.1,
                           distance_threshold=0.1,
                           small_objects_threshold=0,
                           batch_size=1):

    # Tile X_test into overlapping tiles
    X_tiles, tiles_info_X = tile_image_3D(X_test, model_input_shape=input_shape, stride_ratio=stride_ratio)
    
    # Predict on tiles 
    y_pred = model.predict(X_tiles, batch_size=batch_size)

    # Untile predictions
    y_pred = [untile_image_3D(o, tiles_info_X, model_input_shape=input_shape) for o in y_pred]
    
    # Run deep_watershed_3D on untiled predictions
    y_pred = deep_watershed_3D(
        y_pred,
        min_distance=min_distance,
        detection_threshold=detection_threshold,
        distance_threshold=distance_threshold,
        exclude_border=False,
        small_objects_threshold=small_objects_threshold)
        
    # Squeeze out channel dimension from y_true
    y_true = np.squeeze(y_true)
    
    # If batch dimension was squeezed out, add it back in
    if y_true.ndim < 4:
        y_true = np.expand_dims(y_true, 0)

    return y_true, y_pred
