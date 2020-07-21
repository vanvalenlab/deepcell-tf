
"""Functions for handling 3D volumetric data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import shutil
import tempfile

import numpy as np
import scipy.ndimage as nd
import cv2
import scipy.signal

from scipy.ndimage import fourier_shift
from skimage.morphology import ball, disk
from skimage.morphology import binary_erosion
from skimage.feature import register_translation
from skimage import transform

from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import watershed, remove_small_objects
from skimage.segmentation import relabel_sequential

from deepcell_toolbox.utils import erode_edges






def deep_watershed_3D(outputs,
                   min_distance=10,
                   detection_threshold=0.1,
                   distance_threshold=0.01,
                   exclude_border=False,
                   small_objects_threshold=0):
    """Postprocessing function for deep watershed models. Thresholds the inner
    distance prediction to find cell centroids, which are used to seed a marker
    based watershed of the outer distance prediction.

    Args:
        outputs (list): DeepWatershed model output. A list of
            [inner_distance, outer_distance, fgbg].

            - inner_distance: Prediction for the inner distance transform.
            - outer_distance: Prediction for the outer distance transform.
            - fgbg: Prediction for the foregound/background transform.

        min_distance (int): Minimum allowable distance between two cells.
        detection_threshold (float): Threshold for the inner distance.
        distance_threshold (float): Threshold for the outer distance.
        exclude_border (bool): Whether to include centroid detections
            at the border.
        small_objects_threshold (int): Removes objects smaller than this size.

    Returns:
        numpy.array: Uniquely labeled mask.
    """
    inner_distance_batch = outputs[0][:, ..., 0]
    outer_distance_batch = outputs[1][:, ..., 0]

    label_images = []
    for batch in range(inner_distance_batch.shape[0]):
        inner_distance = inner_distance_batch[batch]
        outer_distance = outer_distance_batch[batch]

        coords = peak_local_max(inner_distance,
                                min_distance=min_distance,
                                threshold_abs=detection_threshold,
                                exclude_border=exclude_border)

        markers = np.zeros(inner_distance.shape)
        markers[coords[:, 0], coords[:, 1], coords[:,2]] = 1
        markers = label(markers)
        label_image = watershed(-outer_distance,
                                markers,
                                mask=outer_distance > distance_threshold)
        label_image = erode_edges(label_image, 1)

        # Remove small objects
        label_image = remove_small_objects(label_image, min_size=small_objects_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)

    return label_images


def tile_image_3D(image, model_input_shape=(10, 256, 256), stride_ratio=0.5):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".
    Args:
        image (numpy.array): The 3D image to tile, must be rank 5.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile size
    Returns:
        tuple(numpy.array, dict): An tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).
    Raises:
        ValueError: image is not rank 5.
    """

    # change to 5 dimensions
    if image.ndim != 5:
        raise ValueError('Expected image of 5, got {}'.format(
            image.ndim))

    #
    image_size_z, image_size_x, image_size_y = image.shape[1:4]
    tile_size_z = model_input_shape[0]
    tile_size_x = model_input_shape[1]
    tile_size_y = model_input_shape[2]

    ceil = lambda x: int(np.ceil(x))
    round_to_even = lambda x: int(np.ceil(x / 2.0) * 2)

    #
    stride_z = round_to_even(stride_ratio * tile_size_z)
    stride_x = round_to_even(stride_ratio * tile_size_x)
    stride_y = round_to_even(stride_ratio * tile_size_y)

    if stride_z > tile_size_z:
        stride_z = tile_size_z

    if stride_x > tile_size_x:
        stride_x = tile_size_x

    if stride_y > tile_size_y:
        stride_y = tile_size_y
    #
    rep_number_z = ceil((image_size_z - tile_size_z) / stride_z + 1)
    rep_number_x = ceil((image_size_x - tile_size_x) / stride_x + 1)
    rep_number_y = ceil((image_size_y - tile_size_y) / stride_y + 1)
    new_batch_size = image.shape[0] * rep_number_z * rep_number_x * rep_number_y

    #
    tiles_shape = (new_batch_size, tile_size_z, tile_size_x, tile_size_y, image.shape[4])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    #
    # Calculate overlap of last tile
    overlap_z = (tile_size_z + stride_z * (rep_number_z - 1)) - image_size_z
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    #
    # Calculate padding needed to account for overlap and pad image accordingly
    pad_z = (int(np.ceil(overlap_z / 2)), int(np.floor(overlap_z / 2)))
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0) ##
    padding = (pad_null, pad_z, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, 'constant', constant_values=0)

    #
    counter = 0
    batches = []
    z_starts = []
    z_ends = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_z = []
    overlaps_x = []
    overlaps_y = []
    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                for k in range(rep_number_z):
                    #
                    z_axis = 1
                    x_axis = 2
                    y_axis = 3

                    #
                    # Compute the start and end for each tile
                    if i != rep_number_x - 1:  # not the last one
                        x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                    else:
                        x_start, x_end = image.shape[x_axis] - tile_size_x, image.shape[x_axis]

                    if j != rep_number_y - 1:  # not the last one
                        y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                    else:
                        y_start, y_end = image.shape[y_axis] - tile_size_y, image.shape[y_axis]

                    if k != rep_number_z - 1:  # not the last one
                        z_start, z_end = k * stride_z, k * stride_z + tile_size_z
                    else:
                        z_start, z_end = image.shape[z_axis] - tile_size_z, image.shape[z_axis]

                    # Compute the overlaps for each tile
                    if i == 0:
                        overlap_x = (0, tile_size_x - stride_x)
                    elif i == rep_number_x - 2:
                        overlap_x = (tile_size_x - stride_x, tile_size_x - image.shape[x_axis] + x_end)
                    elif i == rep_number_x - 1:
                        overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                    else:
                        overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                    if j == 0:
                        overlap_y = (0, tile_size_y - stride_y)
                    elif j == rep_number_y - 2:
                        overlap_y = (tile_size_y - stride_y, tile_size_y - image.shape[y_axis] + y_end)
                    elif j == rep_number_y - 1:
                        overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                    else:
                        overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                    if k == 0:
                        overlap_z = (0, tile_size_z - stride_z)
                    elif k == rep_number_z - 2:
                        overlap_z = (tile_size_z - stride_z, tile_size_z - image.shape[z_axis] + z_end)
                    elif k == rep_number_z - 1:
                        overlap_z = ((k - 1) * stride_z + tile_size_z - z_start, 0)
                    else:
                        overlap_z = (tile_size_z - stride_z, tile_size_z - stride_z)

                    tiles[counter] = image[b, z_start:z_end, x_start:x_end, y_start:y_end, :]
                    batches.append(b)
                    x_starts.append(x_start)
                    x_ends.append(x_end)
                    y_starts.append(y_start)
                    y_ends.append(y_end)
                    z_starts.append(z_start)
                    z_ends.append(z_end)
                    overlaps_x.append(overlap_x)
                    overlaps_y.append(overlap_y)
                    overlaps_z.append(overlap_z)
                    counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches

    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['z_starts'] = z_starts
    tiles_info['z_ends'] = z_ends

    tiles_info['overlaps_x'] = overlaps_x
    tiles_info['overlaps_y'] = overlaps_y
    tiles_info['overlaps_z'] = overlaps_z
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['stride_z'] = stride_z
    tiles_info['tile_size_x'] = tile_size_x
    tiles_info['tile_size_y'] = tile_size_y
    tiles_info['tile_size_z'] = tile_size_z

    tiles_info['stride_ratio'] = stride_ratio
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype
    tiles_info['pad_x'] = pad_x
    tiles_info['pad_y'] = pad_y
    tiles_info['pad_z'] = pad_z

    return tiles, tiles_info





def spline_window_3D(window_size, overlap_left, overlap_right, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (scipy.signal.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (scipy.signal.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window


def window_3D(window_size, overlap_z=(5, 5), overlap_x=(32, 32), overlap_y=(32, 32), power=3):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    ####     window_size = (tile_size_z, tile_size_x, tile_size_y)
    window_z = spline_window_3D(window_size[0], overlap_z[0], overlap_z[1], power=power)
    window_x = spline_window_3D(window_size[1], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window_3D(window_size[2], overlap_y[0], overlap_y[1], power=power)

    window_z = np.expand_dims(np.expand_dims(np.expand_dims(window_z, -1), -1), -1)
    window_x = np.expand_dims(np.expand_dims(np.expand_dims(window_x, -1), -1), -1)
    window_y = np.expand_dims(np.expand_dims(np.expand_dims(window_y, -1), -1), -1)

    #print('window_z shape is: ', window_z.shape)
    #print('window_x shape is: ', window_x.shape)
    #print('window_y shape is: ', window_y.shape)
    #print('window_y-t shape is: ', window_y.transpose(1, 0, 2).shape)

    #window = window_x * window_y.transpose(1, 0, 2)
    window = window_z * window_x.transpose(1, 0, 2, 3) * window_y.transpose(1, 2, 0, 3)

    #print('window shape is: ', window.shape)
    #print('')
    return window

def untile_image_3D(tiles, tiles_info, model_input_shape=(10, 256, 256), power=2):

    # Define mininally acceptable tile_size and stride_ratio for spline interpolation
    min_tile_size = 64
    min_tile_height = 0 # TODO - figure out what this should actually be, somewhere between 3 and 10
    min_stride_ratio = 0.5

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']

    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    z_starts = tiles_info['z_starts']
    z_ends = tiles_info['z_ends']

    overlaps_x = tiles_info['overlaps_x']
    overlaps_y = tiles_info['overlaps_y']
    overlaps_z = tiles_info['overlaps_z']
    stride_x = tiles_info['stride_x']
    stride_y = tiles_info['stride_y']
    stride_z = tiles_info['stride_z']

    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    tile_size_z = tiles_info['tile_size_z']
    pad_x = tiles_info['pad_x']
    pad_y = tiles_info['pad_y']
    pad_z = tiles_info['pad_z']

    image_shape = [image_shape[0], image_shape[1], image_shape[2], image_shape[3], tiles.shape[-1]]
    window_size = (tile_size_z, tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=np.float)

    n_tiles = tiles.shape[0]

    for tile, batch, x_start, x_end, y_start, y_end, z_start, z_end, overlap_x, overlap_y, overlap_z in zip(
            tiles, batches, x_starts, x_ends, y_starts, y_ends, z_starts, z_ends, overlaps_x, overlaps_y, overlaps_z):
        if (min_tile_size <= tile_size_x < image_shape[2] and
                min_tile_size <= tile_size_y < image_shape[3] and
                min_tile_height <= tile_size_z and #< image_shape[1] and  # TODO - look at removing the image_shape[1] conditional
                min_stride_ratio <= stride_ratio):

            window = window_3D(window_size, overlap_z=overlap_z, overlap_x=overlap_x, overlap_y=overlap_y, power=power)
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] += tile * window
            #print('tile shape is: ', tile.shape)
            #print('window shape is: ', window.shape)
        else:
            #print('not using window')
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] = tile

        # print('z_start is: ', z_start, ', z_end is: ', z_end, ', tile min is: ', tile.min())
        # print('x_start is: ', x_start, ', x_end is: ', x_end)

    image = image.astype(tiles.dtype)

    x_start = pad_x[0]
    y_start = pad_y[0]
    z_start = pad_z[0]
    x_end = image_shape[2] - pad_x[1]
    y_end = image_shape[3] - pad_y[1]
    z_end = image_shape[1] - pad_z[1]


    image = image[:, z_start:z_end, x_start:x_end, y_start:y_end, :]

    return image

