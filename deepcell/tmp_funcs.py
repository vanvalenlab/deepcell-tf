import os
import cv2
import imageio

import numpy as np
import pandas as pd

from scipy.stats import mode

from skimage.external.tifffile import TiffFile
from skimage.measure import regionprops_table, regionprops

from sklearn.preprocessing import quantile_transform as qt

"""
Data Loading Functions
"""

def get_image(file_name):
    """Read image from file and returns it as a tensor

    Args:
        file_name (str): path to image file

    Returns:
        numpy.array: numpy array of image data
    """
    ext = os.path.splitext(file_name.lower())[-1]
    if ext in ['.tif', '.tiff']:
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))


def load_patient_data(path):
    """Load a single data set
    Args:
        path: Location of the dataset

    Returns:
        np.array: The multiplexed imaging dataset for a specific patient
    """

    channel_list = []

    for _, _, channels in os.walk(path):
        # Sort the channels so we know what order they are in
        channels.sort()

        for channel in channels:
            # Make sure we don't load the segmentations as features!
            if 'Segmentation' not in channel:
                channel_path = os.path.join(path, channel)
                img = get_image(channel_path)
                channel_list.append(img)
    patient_data = np.stack(channel_list, axis=-1)

    return patient_data

def load_mibi_data(path, point_list):
    """Load mibi dataset

    Args:
        path: Path containing the mibi dataset
        point_list: Which patients of the mibi dataset should be loaded
         * point_list = [2, 5, 8, 9, 21, 22, 24, 26, 34, 37, 38, 41]

    Returns:
        np.array: The mibi dataset for all of the points
    """

    # Get the full path for each point
    path_dict = {}
    for _, points, _ in os.walk(path):
        for point in points:
            patient_path = os.path.join(path, point)
            path_dict[point] = patient_path

    mibi_data = []
    
    for point in point_list:

        point = 'Point{}'.format(point)
        full_path = path_dict[point]
        patient_data = load_patient_data(full_path)

        # Make sure dimensions are 2048
        x_dim, y_dim, _ = patient_data.shape
        if (x_dim != 2048) or (y_dim != 2048):
            print(point)
        else:
            mibi_data.append(patient_data)

    return np.array(mibi_data)


def load_celltypes(path, point_list):
    """Load celltype dataset

    Args:
        path: Path containing the cell type dataset
        point_list: Which patients of the mibi/cell dataset should be loaded
         * point_list = [2, 5, 8, 9, 21, 22, 24, 26, 34, 37, 38, 41]

    Returns:
        np.array: The mibi dataset for all of the points
    """
    celltype_images = []
    for point in point_list:
        filename = "P{}_labeledImage.tiff".format(point)
        fullpath = os.path.join(path, filename)
        img = imageio.imread(fullpath)
        celltype_images.append(img)
    celltypes = np.stack(celltype_images, axis=0)
    celltypes = np.expand_dims(celltypes, axis=-1)
    return celltypes

###########################################################################################

"""
Data handeling
"""



def qt_transform(X):
    """Perform sklearn.preprocessing.quantile_transform on batch data

    Args:
        X: data to transform

    Returns:
        np.array: The mibi dataset transformed
    """

    batches = X.shape[-1]
    transformed_data = []

    for batch in range(batches):
        x_batch = X[..., batch]

        x_batch = qt(x_batch,
                     copy=False,
                     output_distribution='uniform',
                     n_quantiles=10)

        transformed_data.append(x_batch)

    return np.array(transformed_data)


def normalize_mibi_data(tmp_mibi_data):#, marker_idx_dict):
    """Normalize mibi data by applying a gaussian smothing on the raw data
       removing bottom 5% of labels and breaking remaining values into quantiles

    Args:
        tmp_mibi_data: raw mibi data loaded via load_mibi func
        marker_idx_dict: look up table to go from index to maker name

    Returns:
        np.array: normalized mibi_data
    """

    num_batches = tmp_mibi_data.shape[0]
    num_channels = tmp_mibi_data.shape[-1]

    mibi_data = []

    for batch in range(num_batches):
        mibi_batch = tmp_mibi_data[batch, ...]
        # channel_data = []

        channel_imgs = []
        channel_th = []
        for channel in range(num_channels):

            batch_channel_data = mibi_batch[..., channel]
            blur_img = cv2.GaussianBlur(batch_channel_data, (3, 3), 0)

            channel_imgs.append(blur_img)

            if len(blur_img[blur_img > 0]) == 0:
#                 print('batch {}'.format(batch), marker_idx_dict[channel])
                channel_th.append(0)
            else:
                low_vals = np.percentile(blur_img[blur_img > 0], 5)
                channel_th.append(low_vals)

        imgs = np.array(channel_imgs).T

        imgs.T[np.tile((np.sum((imgs < channel_th).T, axis=0) == num_channels),
                       (num_channels, 1)).reshape(imgs.T.shape)] = 0

        batch_data = qt_transform(imgs)

        mibi_data.append(batch_data.T)

    return np.array(mibi_data)


###########################################################################################

"""
Auxiliary Functions
"""

def get_marker_dict(path):
    """Load a single dataset

    Args:
        path: Location of the dataset

    Returns:
        np.array: The multiplexed imaging dataset for a specific patient
    """

    marker_list = []

    for _, _, channels in os.walk(path):
        # Sort the channels so we know what order they are in
        channels.sort()
        for channel in channels:
            # Make sure we don't load the segmentations as features!
            if 'Segmentation' not in channel:
                marker_list.append(channel.split('.')[0])

    marker_dict = dict(zip(marker_list, list(range(len(marker_list)))))
    # marker_by_idx_dict = dict(zip(list(range(len(marker_list))),marker_list))

    return marker_dict



# def celltype_to_labeled_img(mibi_celltypes, celltype_data):
#     """
#     DOC STRING
#     """

#     new_label_image = np.zeros(mibi_celltypes.shape)
#     for batch in range(mibi_celltypes.shape[0]):
#         print(batch)
#         props = regionprops(mibi_celltypes[batch, ..., 0])
#         for i, prop in enumerate(props):
#             nli_batch = new_label_image[batch]
#             nli_batch[prop.coords[:, 0], prop.coords[:, 1]] = celltype_data[batch, i]
#         new_label_image[batch] = nli_batch


#     return new_label_image