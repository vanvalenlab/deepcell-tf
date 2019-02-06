"""
metrics.py

Custom error metrics

@author: cpavelchek, msschwartz21
"""
import datetime
import os
import json

import operator

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import skimage.io
import skimage.measure

import pandas as pd
import networkx as nx

from tensorflow.python.platform import tf_logging as logging
import pandas as pd
from sklearn.metrics import confusion_matrix


def im_prep(mask, prediction, win_size):
    """Reads images into ndarrays, and trims them to fit each other"""
    # if trimming not needed, return
    if win_size == 0 or prediction.shape == mask.shape:
        return mask, prediction

    # otherwise, pad prediction to imsize and zero out the outer layer of mask
    mask = mask[win_size:-win_size, win_size:-win_size]
    mask = np.pad(mask, (win_size, win_size), 'constant')
    prediction = np.pad(prediction, (win_size, win_size), 'constant')
    return mask, prediction


def calc_cropped_ious(crop_truth, crop_pred, threshold, iou_matrix):
    """
    Identifies cell objects within a cropped roi that have an IOU above `threshold`
    which results in them being marked as a hit

    Args:
        crop_truth (2D np.array): Cropped numpy array of labeled truth mask
        crop_pred (2D np.array): Cropped numpy array of labeled prediction mask
        threshold (float): Threshold for accepting IOU score as a hit
        iou_matrix (2D np.array): Array for recording hits between predicted and truth objects

    Returns:
        iou_matrix: after updating with any hits found in this crop region
    """
    # for each unique cellID in the given mask...
    for n in np.unique(crop_truth):
        if n == 0:
            continue  # excluding background

        # for each unique cellID in the given prediction...
        for m in np.unique(crop_pred):
            if m == 0:
                continue  # excluding background

            # calculate the intersection over union for pixels in each object
            intersection = np.logical_and(crop_pred == m, crop_truth == n)
            union = np.logical_or(crop_pred == m, crop_truth == n)
            iou = np.sum(intersection) / np.sum(union)

            if iou > threshold:
                iou_matrix[n - 1][m - 1] = 1

    return iou_matrix


def calc_object_ious_fast(y_true, y_pred):
    """
    Identifies cell objects within a cropped roi that have an IOU above `threshold`
    which results in them being marked as a hit

    Args:
        crop_truth (2D np.array): Cropped numpy array of labeled truth mask
        crop_pred (2D np.array): Cropped numpy array of labeled prediction mask

    Returns:
        iou_matrix: after updating with any hits found in this crop region

    Warning:
        Currently does not handle cases in which more than 1 truth and 1 predicted cell ids are
        found in an intersection
    """
    def _joint_or(arr, L):
        '''Calculate overlap of an arr with a list of values'''
        out = arr == L[0]
        for i in L[1:]:
            out = (out) | (arr == i)
        return out

    # Initialize iou matrix
    # Add third dimension to seperate merges
    iou_matrix = np.zeros((y_true.max(), y_pred.max(), 2))

    # Find an intersection mask of all regions of intersection
    mask = np.logical_or(y_true != 0, y_pred != 0)
    mask_lbl = skimage.measure.label(mask, connectivity=2)

    # Loop over each region of intersection
    for i in np.unique(mask_lbl):
        if i == 0:
            continue  # exclude background

        # Extract cell ids from y_pred and y_true
        tid = np.unique(y_true[mask_lbl == i])
        pid = np.unique(y_pred[mask_lbl == i])

        # First handle cases when there are only two cell ids
        if (len(tid) == 1) & (len(pid) == 1):
            intersection = np.logical_and(y_true == tid[0], y_pred == pid[0])
            union = np.logical_or(y_true == tid[0], y_pred == pid[0])
            iou = np.sum(intersection) / np.sum(union)

            iou_matrix[tid - 1, pid - 1, 0] = iou

        else:


<< << << < HEAD
            # intersection = np.logical_and(
            #     _joint_or(y_true, tid), _joint_or(y_pred, pid))
            # union = np.logical_or(_joint_or(y_true, tid),
            #                       _joint_or(y_pred, pid))
            # iou = np.sum(intersection) / np.sum(union)

            t_error, p_error = [], []
== == == =
            intersection = np.logical_and(
                _joint_or(y_true, tid), _joint_or(y_pred, pid))
            union = np.logical_or(_joint_or(y_true, tid),
                                  _joint_or(y_pred, pid))
            iou = np.sum(intersection) / np.sum(union)
>>>>>> > 0dbcbf8b403d816bf12451617fb7e84229662a62

            for t in tid:
                for p in pid:
                    intersection = np.logical_and(y_true == t, y_pred == p)
                    union = np.logical_or(y_true == t, y_pred == p)
                    iou = np.sum(intersection) / np.sum(union)
                    iou_matrix[t - 1, p - 1, 1] = iou

    return iou_matrix


# def get_iou_matrix_quick(y_true, y_pred, crop_size, threshold=0.5):
#     """Calculate Intersection-Over-Union Matrix for ground truth and predictions
#     based on object labels

#     Intended to work on 2D arrays,
#     but placing this function in a loop could extend to 3D or higher

#     Arguments:
#         pred (2D np.array): predicted, labeled mask
#         truth (2D np.array): ground truth, labeled mask
#         crop_size (int): Cropping images is faster to calculate but less accurate
#         threshold (:obj:`float`, optional): If IOU is above threshold,
#            cells are considered overlapping, default 0.5

#     Returns:
#         iou_matrix: 1 indicates an object pair with an IOU score above threshold

#     Warning:
#         Currently non-functional because cropping functionality needs to be restored.
#     """

#     # Setup empty iou matrix based on number of true and predicted cells
#     iou_matrix = np.zeros((y_true.max(), y_pred.max()))
#     print(iou_matrix.shape)
#     print('true', y_true.max(), 'pred', y_pred.max())

#     # Get image size parameters, assumes 2D inputs
#     x_size = y_true.shape[0]
#     y_size = y_true.shape[1]

#     # Crop input images and calculate the iou's for the cells present
#     # Updates iou_matrix value during each loop
#     # Consider using np.split as a potentially faster alternative
#     for x in range(0, x_size, crop_size):
#         for y in range(0, y_size, crop_size):
#             crop_pred = y_pred[x:x + crop_size, y:y + crop_size]
#             crop_truth = y_true[x:x + crop_size, y:y + crop_size]
#             # iou_matrix = calc_cropped_ious(crop_truth, crop_pred, threshold, iou_matrix)
#             iou_matrix = calc_object_ious_fast(
#                 crop_truth, crop_pred, threshold)

#     return iou_matrix


def get_dice_jaccard(iou_matrix):
    """Caclulates DICE score for object based metrics
    # Arguments:
        iou_matrix: Matrix of Intersection over Union
    # Returns
        dice: DICE score for object based
        jaccard: Jaccard score for object based
    """
    iou_sum = np.sum(iou_matrix)
    pred_max = iou_matrix.shape[1] - 1
    truth_max = iou_matrix.shape[0] - 1

    dice = 2 * iou_sum / (2 * iou_sum + pred_max -
                          iou_sum + truth_max - iou_sum)
    jaccard = dice / (2 - dice)

    return dice, jaccard


def reshape_padded_tiled_2d(arr):
    """Takes in a 3 or 4d stack and reshapes so that arrays from the zeroth axis are tiled in 2D

    Args:
        arr (np.array): 3 or 4D array to be reshaped with reshape axis as zeroth axis

    Returns:
        np.array: Output array should be two dimensional except for a possible channel dimension

    Raises:
        ValueError: Only accepts 3 or 4D input arrays
    """
    # Check if input is 3 or 4 dimensions for padding
    # Add border of zeros around input arr
    if len(arr.shape) == 4:
        pad = np.zeros((arr.shape[0],
                        arr.shape[1] + 2,
                        arr.shape[2] + 2,
                        arr.shape[3]))
    elif len(arr.shape) == 3:
        pad = np.zeros((arr.shape[0],
                        arr.shape[1] + 2,
                        arr.shape[2] + 2))
    else:
        raise ValueError('Only supports input of dimensions 3 or 4. '
                         'Array of dimension {} received as input'.format(
                             len(arr.shape)))

    # Add data into padded array
    pad[:, 1:-1, 1:-1] = arr

    # Split array into list of as many arrays as are in zeroth dimension
    splitlist = np.split(pad, pad.shape[0], axis=0)

    # Concatenate into single 2D array
    out = np.concatenate(splitlist, axis=2)

    return out


def stats_objectbased(y_true,
                      y_pred,
                      object_threshold=0.5,
                      ndigits=4,
                      crop_size=None,
                      return_iou=False):
    """
    Calculate summary statistics (DICE/Jaccard index and confusion matrix)
    for a labeled images on a per-object basis

    Args:
        y_true (3D np.array): Labled ground truth annotations (batch,x,y)
        y_pred (3D np.array): Labeled predictions (batch,x,y)
        object_threshold (:obj:`float`, optional): Sets criteria for jaccard index
            to declare object overlap
        ndigits (:obj:`int`, optional): Sets number of digits for rounding, default 4
        crop_size (:obj:`int`, optional): Enables cropping for object calculations, default None
        return_iou (:obj:`bool`, optional): Returns iou_matrix if True, default False

    Returns:
        iou_matrix: np.array containing iou scores

    Raises:
        ValueError: If y_true and y_pred are not the same shape
        ValueError: If cropping specified, because cropping not currently available
    """

    def _round(x):
        return round(x, ndigits)

    if y_pred.shape != y_true.shape:
        raise ValueError('Shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of y_true is: {}'.format(
                             y_pred.shape, y_true.shape))

    # Reshape to be tiled 2D image
    y_true = reshape_padded_tiled_2d(y_true[:, :, :])
    y_pred = reshape_padded_tiled_2d(y_pred[:, :, :])

    # Calculate labels using skimage so labels unique across entire tile
    y_true = skimage.measure.label(y_true, connectivity=2)
    y_pred = skimage.measure.label(y_pred, connectivity=2)

    # Calculate iou matrix on reshaped, masked arrays
    if crop_size is not None:
        raise ValueError('Cropping functionality is not currently available.')
        # iou_matrix = get_iou_matrix_quick(
        #     y_true, y_pred, crop_size, threshold=object_threshold)
    else:
        iou_matrix = calc_object_ious_fast(y_true, y_pred)

    # Get performance stats
    stats = calc_2d_object_stats((iou_matrix > object_threshold).astype('int'))

    false_pos_perc_err = stats['false_pos'] / \
        (stats['false_pos'] + stats['false_neg'])
    false_neg_perc_err = stats['false_neg'] / \
        (stats['false_pos'] + stats['false_neg'])

    false_pos_perc_pred = stats['false_pos'] / stats['pred_cells']
    false_neg_perc_truth = stats['false_neg'] / stats['true_cells']

    perc_merged = stats['merge'] / stats['true_cells']
    perc_divided = stats['split'] / stats['true_cells']

    acc = (stats['pred_cells'] - stats['false_pos']) / stats['true_cells']

    print('\n____________________Object-based statistics____________________\n')
    print('Intersection over Union thresholded at {} for object detection'.format(
        object_threshold))
    print('Dice/F1 index: {}\nJaccard index: {}'.format(
        _round(stats['dice']), _round(stats['jaccard'])))
    print('Number of cells predicted:', stats['pred_cells'])
    print('Number of cells present in ground truth:', stats['true_cells'])
    print('Accuracy: {}%\n'.format(_round(acc * 100)))

    print('#true positives: {}'.format(_round(stats['pred_true_pos'])))

    print('#false positives: {}\t% of total error: {}\t% of predicted incorrect: {}'.format(
        _round(stats['false_pos']),
        _round(false_pos_perc_err * 100),
        _round(false_pos_perc_pred * 100)))

    print('#false negatives: {}\t% of total error: {}\t% of ground truth missed: {}'.format(
        _round(stats['false_neg']),
        _round(false_neg_perc_err * 100),
        _round(false_neg_perc_truth * 100)))

    print('#incorrect merges: {}\t% of ground truth merged: {}'.format(
        stats['merge'], _round(perc_merged * 100)))
    print('#incorrect divisions: {}\t% of ground truth divided: {}'.format(
        stats['split'], _round(perc_divided * 100)))

    if return_iou:
        return iou_matrix


def calc_2d_object_stats(iou_matrix):
    """Calculates basic statistics to evaluate classification accuracy for a 2d image

    Args:
        iou_matrix (np.array): 2D array with dimensions (#true_cells,#predicted_cells)

    Returns:
        dict: Dictionary containing all statistics computed by function
    """
    true_cells = iou_matrix.shape[0]
    pred_cells = iou_matrix.shape[1]

    # Calculate values based on projecting along prediction axis
    pred_proj = iou_matrix.sum(axis=1)
    # Zeros (aka absence of hits) correspond to true cells missed by prediction
    false_neg = np.count_nonzero(pred_proj == 0)
    # More than 2 hits corresponds to true cells hit twice by prediction, aka split
    split = np.count_nonzero(pred_proj >= 2)

    # Calculate values based on projecting along truth axis
    truth_proj = iou_matrix.sum(axis=0)
    # Empty hits indicate predicted cells that do not exist in true cells
    false_pos = np.count_nonzero(truth_proj == 0)
    # More than 2 hits indicates more than 2 true cells corresponding to 1 predicted cell
    merge = np.count_nonzero(truth_proj >= 2)

    # Ones are true positives excluding merge errors
    true_pos = np.count_nonzero(pred_proj == 1) - \
        (truth_proj[truth_proj >= 2].sum())

    # Calc dice jaccard stats for objects
    dice, jaccard = get_dice_jaccard(iou_matrix)

    return {
        'true_cells': true_cells,
        'pred_cells': pred_cells,
        'false_neg': false_neg,
        'split': split,
        'true_pos': true_pos,
        'false_pos': false_pos,
        'merge': merge,
        'dice': dice,
        'jaccard': jaccard
    }


def calc_2d_object_stats_3diou(iou_matrix):
    """Calculates basic statistics to evaluate classification accuracy for a 2d image

    Args:
        iou_matrix (np.array): 2D array with dimensions (#true_cells,#predicted_cells)

    Returns:
        dict: Dictionary containing all statistics computed by function
    """
    true_cells = iou_matrix.shape[0]
    pred_cells = iou_matrix.shape[1]

    iou = iou_matrix[:, :, 0]

    # Calculate values based on projecting along prediction axis
    pred_proj = iou.sum(axis=1)
    # Zeros (aka absence of hits) correspond to true cells missed by prediction
    false_neg = np.count_nonzero(pred_proj == 0)

    # Calculate values based on projecting along truth axis
    truth_proj = iou.sum(axis=0)
    # Empty hits indicate predicted cells that do not exist in true cells
    false_pos = np.count_nonzero(truth_proj == 0)

    # Ones are true positives excluding merge errors
    true_pos = np.count_nonzero(pred_proj == 1)

    # Deal with merges
    iou = (iou_matrix[:, :, 1] != 0).astype('int')
    pred_proj = iou.sum(axis=1)
    truth_proj = iou.sum(axis=0)

    # More than 2 hits corresponds to true cells hit twice by prediction, aka split
    # split = np.count_nonzero(pred_proj >= 2)
    # More than 2 hits indicates more than 2 true cells corresponding to 1 predicted cell
    # merge = np.count_nonzero(truth_proj >= 2)

    # # Calc dice jaccard stats for objects
    # dice, jaccard = get_dice_jaccard(iou_matrix)

    return {
        'true_cells': true_cells,
        'pred_cells': pred_cells,
        'false_neg': false_neg,
        'split': split,
        'true_pos': true_pos,
        'false_pos': false_pos,
        'merge': merge,
        'dice': dice,
        'jaccard': jaccard
    }


def stats_pixelbased(y_true, y_pred, ndigits=4, return_stats=False):
    """Calculates pixel-based statistics (Dice, Jaccard, Precision, Recall, F-measure)

    Takes in raw prediction and truth data. Applies labeling to prediction
        before calculating statistics.

    Args:
        y_true (3D np.array): Binary ground truth annotations for a single feature, (batch,x,y)
        y_pred (3D np.array): Binary predictions for a single feature, (batch,x,y)
        ndigits (:obj:`int`, optional): Sets number of digits for rounding, default 4
        return_stats (:obj:`bool`, optional): Returns dictionary of statistics, default False

    Returns:
        dictionary: optionally returns a dictionary of statistics

    Raises:
        ValueError: Shapes of `y_true` and `y_pred` do not match.

    Warning:
        Comparing labeled to unlabeled data will produce very low accuracy scores.
        Make sure to input the same type of data for `y_true` and `y_pred`

    Todo:
        Should `y_true` be transformed to match `y_pred` or vice versa
    """

    def _round(x):
        return round(x, ndigits)

    if y_pred.shape != y_true.shape:
        raise ValueError('Shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of y_true is: {}'.format(
                             y_pred.shape, y_true.shape))

    pred = y_pred
    truth = y_true

    if pred.sum() == 0 and truth.sum() == 0:
        logging.warning('DICE score is technically 1.0, '
                        'but prediction and truth arrays are empty. ')

    # Calculations for IOU
    intersection = np.logical_and(pred, truth)
    union = np.logical_or(pred, truth)

    # Sum gets count of positive pixels
    dice = (2 * intersection.sum() / (pred.sum() + truth.sum()))
    jaccard = intersection.sum() / union.sum()

    print('\n____________________Pixel-based statistics____________________\n')
    print('Dice: {}\nJaccard: {}\n'.format(
        _round(dice), _round(jaccard)))

    precision = intersection.sum() / pred.sum()
    recall = intersection.sum() / truth.sum()
    Fmeasure = (2 * precision * recall) / (precision + recall)
    print('Precision: {}\nRecall: {}\nF-measure: {}'.format(
        _round(precision), _round(recall), _round(Fmeasure)))

    if return_stats:
        return {
            'dice': dice,
            'jaccard': jaccard,
            'precision': precision,
            'recall': recall,
            'Fmeasure': Fmeasure
        }


class ObjectAccuracy:
    def __init__(self, y_true, y_pred, cutoff1, cutoff2):

        self.y_true = y_true
        self.y_pred = y_pred
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2

        self.n_true = y_true.max()
        self.n_pred = y_pred.max()
        self.n_obj = self.n_true + self.n_pred

        # Initialize error counters
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.merge = 0
        self.split = 0

        # Check if either frame is empty before proceeding
        if 0 not in [self.n_true, self.n_pred]:
            self.calc_iou()
            self.make_matrix()
            self.linear_assignment()
            self.assign_loners()
            self.array_to_graph()
            self.classify_graph()

    def calc_iou(self):

        self.iou = np.zeros((self.n_true, self.n_pred))

        # Make all pairwise comparisons to calc iou
        for t in np.unique(self.y_true)[1:]:  # skip 0
            for p in np.unique(self.y_pred)[1:]:  # skip 0
                intersection = np.logical_and(
                    self.y_true == t, self.y_pred == p)
                union = np.logical_or(self.y_true == t, self.y_pred == p)
                # Subtract 1 from index to account for skipping 0
                self.iou[t - 1, p - 1] = intersection.sum() / union.sum()

    def make_matrix(self):

        self.cm = np.ones((self.n_obj, self.n_obj))

        # Assign 1 - iou to top left and bottom right
        self.cm[:self.n_true, :self.n_pred] = 1 - self.iou
        self.cm[-self.n_pred:, -self.n_true:] = 1 - self.iou.T

        # Calculate diagonal corners
        bl = self.cutoff1 * \
            np.eye(self.n_pred) + np.ones((self.n_pred, self.n_pred)) - \
            np.eye(self.n_pred)
        tr = self.cutoff1 * \
            np.eye(self.n_true) + np.ones((self.n_true, self.n_true)) - \
            np.eye(self.n_true)

        # Assign diagonals to cm
        self.cm[-self.n_pred:, :self.n_pred] = bl
        self.cm[:self.n_true, -self.n_true:] = tr

    def linear_assignment(self):

        self.results = linear_sum_assignment(self.cm)

        # Map results onto cost matrix
        self.cm_res = np.zeros(self.cm.shape)
        self.cm_res[self.results[0], self.results[1]] = 1

        self.match = np.where(self.cm_res[self.n_true:, self.n_pred:] == 1)
        self.true_pos += len(self.match[0])

        self.loners_pred, _ = np.where(
            self.cm_res[-self.n_pred:, :self.n_pred] == 1)
        self.loners_true, _ = np.where(
            self.cm_res[:self.n_true, -self.n_true:] == 1)

    def plot_assignment_matrix(self):

        fig, ax = plt.subplots()
        ax.imshow(self.cm_res)
        plt.axhline(self.n_true-0.5, c='r')
        plt.axvline(self.n_pred-0.5, c='r')

    def plot_true_positives(self):

        for ti, pi in zip(self.match[0], self.match[1]):
            fig, ax = plt.subplots(1, 2)
            # Shift index by 1 b/c skipped 0
            ax[0].imshow(self.y_true == (ti+1))
            ax[1].imshow(self.y_pred == (pi+1))

    def plot_loners(self):

        def _joint_or(arr, L):
            '''Calculate overlap of an arr with a list of values'''
            out = arr == L[0]
            for i in L[1:]:
                out = (out) | (arr == i)
            return out

        fig, ax = plt.subplots(1, 2)
        # Shift index by 1 b/c skipped 0
        ax[0].imshow(_joint_or(self.y_true, self.loners_true+1))
        ax[1].imshow(_joint_or(self.y_pred, self.loners_pred+1))

        # if len(self.loners_true)>1:
        #     ax[0].imshow(_joint_or(self.y_true,self.loners_true+1))
        # elif len(self.loners_true)==1:
        #     ax[0].imshow(self.y_true==self.loners_true[0]+1)

        # if len(self.loners_pred)>1:
        #     ax[1].imshow(_joint_or(self.y_pred,self.loners_pred+1))
        # elif self.loners_pred.shape[0]==1:
        #     ax[1].imshow(self.y_pred==self.loners_pred[0]+1)

    def assign_loners(self):

        self.n_pred2 = len(self.loners_pred)
        self.n_true2 = len(self.loners_true)
        self.n_obj2 = self.n_pred2 + self.n_true2

        self.cost_l = np.zeros((self.n_true2, self.n_pred2))

        for i, t in enumerate(self.loners_true):
            for j, p in enumerate(self.loners_pred):
                self.cost_l[i, j] = self.iou[t, p]

        self.cost_l_bin = self.cost_l > self.cutoff2

    def array_to_graph(self):

        # Use meshgrid to get true and predicted cell index for each val
        tt, pp = np.meshgrid(np.arange(self.cost_l_bin.shape[0]), np.arange(
            self.cost_l_bin.shape[1]), indexing='ij')

        df = pd.DataFrame({
            'true': tt.flatten(),
            'pred': pp.flatten(),
            'weight': self.cost_l_bin.flatten()
        })

        # Change cell index to str names
        df['true'] = 'true_' + df['true'].astype('str')
        df['pred'] = 'pred_' + df['pred'].astype('str')
        nodes = list(df['true'].unique())+list(df['pred'].unique())

        # Drop 0 weights to only retain overlapping cells
        dfedge = df.drop(df[df['weight'] == 0].index)

        # Create graph from edges
        self.G = nx.from_pandas_edgelist(dfedge, source='true', target='pred')

        # Add nodes to ensure all cells are included
        self.G.add_nodes_from(nodes)

    def classify_graph(self):

        # Find subgraphs, e.g. merge/split
        for g in nx.connected_component_subgraphs(self.G):
            k = max(dict(g.degree).items(), key=operator.itemgetter(1))[0]

            # Process isolates first
            if g.degree[k] == 0:
                if 'pred' in k:
                    self.false_pos += 1
                elif 'true' in k:
                    self.false_neg += 1
            # Eliminate anything with max degree 1
            # Aka true pos
            elif g.degree[k] == 1:
                self.true_pos += 1
            # Process merges and split
            else:
                if 'pred' in k:
                    self.merge += 1
                elif 'true' in k:
                    self.split += 1

    def print_report(self):

        print('Number of cells predicted:', self.n_pred)
        print('Number of true cells:', self.n_true)
        print('True positives: {}, Accuracy: {}'.format(
            self.true_pos, np.round(self.true_pos/self.n_true, 2)
        ))
        print('False positives: {}'.format(self.false_pos))
        print('False negatives: {}'.format(self.false_neg))
        print('Merges: {}'.format(self.merge))
        print('Splits: {}'.format(self.split))

class Metrics:
    '''
    Class to facilitate calculating and saving various classification metrics

    Args:
        model_name (str): Name of the model which determines output file names
        outdir (:obj:`str`, optional): Directory to save json file, default ''
        object_threshold (:obj:`float`, optional): Sets criteria for jaccard index
            to declare object overlap
        pixel_threshold (:obj:`float`, optional): Threshold for converting predictions to binary
        ndigits (:obj:`int`, optional): Sets number of digits for rounding, default 4
        crop_size (:obj:`int`, optional): Enables cropping for object calculations, default None
        return_iou (:obj:`bool`, optional): Returns iou_matrix if True, default False
        feature_key (:obj:`list`, optional): List of strings to use as feature names
    '''

    def __init__(self, model_name,
                 outdir='',
                 object_threshold=0.5,
                 pixel_threshold=0.5,
                 ndigits=4,
                 crop_size=None,
                 return_iou=False,
                 feature_key=[]):

        self.model_name = model_name
        self.outdir = outdir
        self.object_threshold = object_threshold
        self.pixel_threshold = pixel_threshold
        self.ndigits = ndigits
        self.crop_size = crop_size
        self.return_iou = return_iou
        self.feature_key = feature_key

        # Initialize output list to collect stats
        self.output = []

    def all_pixel_stats(self, y_true, y_pred):
        '''Collect pixel statistics for each feature.

        y_true should have the appropriate transform applied to match y_pred

        Args:
            y_true (4D np.array): Ground truth annotations after application of transform
            y_pred (4D np.array): Model predictions without labeling

        Raises:
            ValueError: If y_true and y_pred are not the same shape
        '''

        if y_pred.shape != y_true.shape:
            raise ValueError('Shape of inputs need to match. Shape of prediction '
                             'is: {}.  Shape of y_true is: {}'.format(
                                 y_pred.shape, y_true.shape))

        n_features = y_pred.shape[-1]

        # Intialize df to collect pixel stats
        self.pixel_df = pd.DataFrame()

        # Set numeric feature key if existing key is not write length
        if n_features != len(self.feature_key):
            self.feature_key = range(n_features)

        for i, k in enumerate(self.feature_key):
            print('\nChannel', k)
            yt = y_true[:, :, :, i] > self.pixel_threshold
            yp = y_pred[:, :, :, i] > self.pixel_threshold
            stats = stats_pixelbased(
                yt, yp, ndigits=self.ndigits, return_stats=True)
            self.pixel_df = self.pixel_df.append(
                pd.DataFrame(stats, index=[k]))

        # Save stats to output dictionary
        self.output = self.output + self.df_to_dict(self.pixel_df)

        # Calculate confusion matrix
        cm = self.calc_confusion_matrix(y_true, y_pred)
        self.output.append(dict(
            name='confusion_matrix',
            value=cm.tolist(),
            feature='all',
            stat_type='pixel'
        ))

    def df_to_dict(self, df):
        """Output pandas df as a list of dictionary objects

        Args:
            df (pd.DataFrame): Dataframe of statistics for each channel

        Returns:
            list: List of dictionaries
        """

        # Initialize output dictionary
        L = []

        # Write out average statistics
        for k, v in df.mean().iteritems():
            L.append(dict(
                name=k,
                value=v,
                feature='average',
                stat_type='pixel'
            ))

        # Save individual stats to list
        for i, row in df.iterrows():
            for k, v in row.iteritems():
                L.append(dict(
                    name=k,
                    value=v,
                    feature=i,
                    stat_type='pixel'
                ))

        return L

    def calc_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix for pixel classification data.

        Args:
            y_true (4D np.array): Ground truth annotations after any necessary transformations
            y_pred (4D np.array): Prediction array

        Returns:
            confusion_matrix: nxn array determined by number of features
        """

        # Argmax collapses on feature dimension to assign class to each pixel
        # Flatten is requiremed for confusion matrix
        y_true = y_true.argmax(axis=-1).flatten()
        y_pred = y_pred.argmax(axis=-1).flatten()

        return confusion_matrix(y_true, y_pred)

    def calc_object_stats(self, y_true, y_pred):
        """Calculate object statistics and save to output

        Args:
            y_true (3D np.array): Labeled ground truth annotations
            y_pred (3D np.array): Labeled prediction mask
        """
        self.iou_matrix = stats_objectbased(y_true, y_pred,
                                            object_threshold=self.object_threshold,
                                            ndigits=self.ndigits,
                                            crop_size=self.crop_size,
                                            return_iou=True)

        # Get stats dictionary
        stats = calc_2d_object_stats(self.iou_matrix)
        for k, v in stats.items():
            self.output.append(dict(
                name=k,
                value=v,
                feature='object',
                stat_type='object'
            ))

    def run_all(self,
                y_true_lbl,
                y_pred_lbl,
                y_true_unlbl,
                y_pred_unlbl):
        """Runs pixel and object base statistics and ouputs to file

        Args:
            y_true_lbl (3D np.array): Labeled ground truth annotation, (sample,x,y)
            y_pred_lbl (3D np.array): Labeled prediction mask, (sample,x,y)
            y_true_unlbl (4D np.array): Ground truth annotation after necessary transforms,
                (sample,x,y,feature)
            y_pred_unlbl (4D np.array): Predictions, (sample,x,y,feature)
        """

        print('Starting pixel based statistics')
        self.all_pixel_stats(y_true_unlbl, y_pred_unlbl)

        print('Starting object based statistics')
        self.calc_object_stats(y_true_lbl, y_pred_lbl)

        self.save_to_json(self.output)

    def save_to_json(self, L):
        """Save list of dictionaries to json file with file metadata

        Args:
            L (list): List of metric dictionaries
        """

        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        outname = os.path.join(
            self.outdir, self.model_name+'_'+todays_date+'.json')

        # Configure final output
        D = {}

        # Record metadata
        D['metadata'] = dict(
            model_name=self.model_name,
            date=todays_date
        )

        # Record metrics
        D['metrics'] = L

        with open(outname, 'w') as outfile:
            json.dump(D, outfile)

        print('Saved to', outname)
