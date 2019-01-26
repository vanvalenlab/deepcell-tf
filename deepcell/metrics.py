"""
metrics.py

Custom error metrics

@author: cpavelchek
"""

import numpy as np
import skimage.io
import skimage.measure
from tensorflow.python.platform import tf_logging as logging

from deepcell.image_generators import _transform_masks


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


def get_iou_matrix_quick(y_true, y_pred, threshold, crop_size, im_size):
    """Calculate Intersection-Over-Union Matrix for ground truth and predictions
    based on object labels

    Intended to work on 2D arrays, but placing this function in a loop could extend to 3D or higher

    Arguments
        pred (2D np.array): predicted, unlabeled mask
        truth (2D np.array): ground truth, unlabeled mask
        threshold (float): If IOU is above threshold, cells are considered overlapping
        crop_size (int): Cropping images is faster to calculate but less accurate
        im_size (int): Original image size.  (Assumes square images).

    Returns
        iou_matrix: 1 indicates an object pair with an IOU score above threshold
    """
    # Calculate labels using skimage
    y_true = skimage.measure.label(y_true,connectivity=2)
    y_pred = skimage.measure.label(y_pred,connectivity=2)

    # Setup empty iou matrix based on number of true and predicted cells
    iou_matrix = np.zeros(y_true.max(),y_pred.max())

    # Crop input images and calculate the iou's for the cells present
    # Updates iou_matrix value during each loop
    for x in range(0, im_size, crop_size):
        for y in range(0, im_size, crop_size):
            crop_pred = y_pred[x:x + crop_size, y:y + crop_size]
            crop_truth = y_true[x:x + crop_size, y:y + crop_size]
            iou_matrix = calc_cropped_ious(crop_truth, crop_pred, threshold, iou_matrix)
    
    return iou_matrix


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

    dice = 2 * iou_sum / (2 * iou_sum + pred_max - iou_sum + truth_max - iou_sum)
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
                        arr.shape[1]+2,
                        arr.shape[2]+2, 
                        arr.shape[3]))
    elif len(arr.shape) == 3:
        pad = np.zeros((arr.shape[0],
                        arr.shape[1]+2,
                        arr.shape[2]+2))
    else:
        raise ValueError('Only supports input of dimensions 3 or 4. Array of dimension {} received as input'.format(
            len(arr.shape)))

    # Add data into padded array
    pad[:,1:-1,1:-1] = arr

    # Split array into list of as many arrays as are in zeroth dimension
    splitlist = np.split(pad, pad.shape[0], axis=0)

    # Concatenate into single 2D array
    out = np.concatenate(splitlist, axis=2)

    return(out)


def stats_objectbased(y_true,
                      y_pred,
                      transform=None,
                      channel_index=0,
                      dice_iou_threshold=.5,
                      merge_iou_threshold=1e-5,
                      ndigits=4,
                      crop_size=32,
                      im_size=2048):
    """
    Calculate summary statistics (DICE/Jaccard index and confusion matrix)
    for a single channel on a per-object basis

    `y_true` and `y_pred` should be the same shape after applying `transform`
    to `y_true`. `channel_index` selects a single channel from `y_true` and `y_pred`
    to use for stat calculations. Relies on  `skimage.measure.label` to define cell
    objects which are used to calculate stats

    Args:
        y_true (4D np.array): Ground truth annotations for a single channel (batch,x,y,channel)
        y_pred (4D np.array): Predictions for a single channel (batch,x,y,channel)
        transform (:obj:`str`, optional): Applies a transformation to y_true, default None
        channel_index (:obj:`int`, optional): Selects channel to compare for object stats, default 0
        dice_iou_threshold (:obj:`float`, optional): default, 0.5
        merge_iou_threshold (:obj:`float`, optional): default, 1e-5
        ndigits (:obj:`int`, optional): Sets number of digits for rounding, default 4
        crop_size (:obj:`int`, optional): default 32

    Raises:
        ValueError: If y_true and y_pred are not the same shape
    """

    def _round(x):
        return round(x, ndigits)

    # Apply transformation if requested
    if transform is not None:
        y_true = _transform_masks(y_true, transform)

    if y_pred.shape != y_true.shape:
        raise ValueError('Shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of y_true after transform is: {}'.format(
                             y_pred.shape, y_true.shape))

    # Loop over each batch sample to process
    for i in range(y_true.shape[0]):
        stats_iou_matrix[i] = get_iou_matrix_quick(
            y_true[i], y_pred[i], dice_iou_threshold, crop_size, im_size)

    dice, jaccard = get_dice_jaccard(stats_iou_matrix)

    # Calculate false negative/positive rates
    false_negatives = 0
    for n in range(stats_iou_matrix.shape[0]):
        if stats_iou_matrix[n, :].sum() == 0:
            false_negatives += 1

    false_positives = 0
    for m in range(stats_iou_matrix.shape[1]):
        if stats_iou_matrix[:, m].sum() == 0:
            false_positives += 1

    false_pos_perc_err = false_positives / (false_positives + false_negatives)
    false_neg_perc_err = false_negatives / (false_positives + false_negatives)

    false_pos_perc_pred = false_positives / stats_iou_matrix.shape[1]
    false_neg_perc_truth = false_negatives / stats_iou_matrix.shape[0]

    # Calculate merge/division error rates
    merge_div_iou_matrix = get_iou_matrix_quick(
        y_true, y_pred, merge_iou_threshold, crop_size)

    divided = 0
    for n in range(merge_div_iou_matrix.shape[0]):
        overlaps = merge_div_iou_matrix[n, :].sum()
        if overlaps > 1:
            divided += overlaps - 1

    merged = 0
    for m in range(merge_div_iou_matrix.shape[1]):
        overlaps = merge_div_iou_matrix[:, m].sum()
        if overlaps > 1:
            merged += overlaps - 1

    perc_merged = merged / stats_iou_matrix.shape[0]
    perc_divided = divided / stats_iou_matrix.shape[0]

    acc = (stats_iou_matrix.shape[1] - false_positives) / stats_iou_matrix.shape[0]
    acc = _round(100 * acc)

    print('\n____________________Object-based statistics____________________\n')
    print('Intersection over Union thresholded at:', dice_iou_threshold)
    print('dice/F1 index: {}\njaccard index: {}'.format(
        _round(dice), _round(jaccard)))
    print('Number of cells predicted:', stats_iou_matrix.shape[1])
    print('Number of cells present in ground truth:', stats_iou_matrix.shape[0])
    print('Accuracy: {}%\n'.format(acc))

    print('#false positives: {}\t% of total error: {}\t% of predicted incorrect: {}'.format(
        _round(false_positives),
        _round(false_pos_perc_err),
        _round(false_pos_perc_pred)))

    print('#false negatives: {}\t% of total error: {}\t% of ground truth missed: {}'.format(
        _round(false_negatives),
        _round(false_neg_perc_err),
        _round(false_neg_perc_truth)))

    print('\nIntersection over Union thresholded at:', merge_iou_threshold)
    print('#incorrect merges: {}\t% of ground truth merged: {}'.format(
        merged, _round(perc_merged)))
    print('#incorrect divisions: {}\t% of ground truth divided: {}'.format(
        divided, _round(perc_divided)))

    return(stats_iou_matrix)

def calc_2d_object_stats(iou_matrix,return_dict=False):
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
    truth_true_pos = np.count_nonzero(pred_proj==1)
    false_neg = np.count_nonzero(pred_proj==0)
    split = np.count_nonzero(pred_proj>=2)

    # Calculate values based on projecting along truth axis
    truth_proj = iou_matrix.sum(axis=0)
    pred_true_pos = np.count_nonzero(truth_proj==1)
    false_pos = np.count_nonzero(truth_proj==0)
    merge = np.count_nonzero(truth_proj>=2)

    return {
        'true_cells':true_cells,
        'pred_cells':pred_cells,
        'truth_true_pos':truth_true_pos,
        'false_neg':false_neg,
        'split':split,
        'pred_true_pos':pred_true_pos,
        'false_pos':false_pos,
        'merge':merge
    }

def stats_pixelbased(y_true, y_pred, transform=None, channel_index=0, threshold=0.5, ndigits=4, return_stats=False):
    """Calculates pixel-based statistics (Dice, Jaccard, Precision, Recall, F-measure)

    Takes in raw prediction and truth data. Applies a transformation to truth data. Before calculating statistics.

    Args:
        y_true (4D np.array): Raw ground truth annotations, (batch,x,y,channel)
        y_pred (np.array): Raw predictions, (batch,x,y,channel)
        transform (:obj:`str`, optional): Applies a transformation to y_true, default None
        channel_index (:obj:`int`, optional): Selects channel to compare for object stats, default 0
        threshold (:obj:`float`, optional): Threshold to use on prediction data to make binary
        ndigits (:obj:`int`, optional): Sets number of digits for rounding, default 4
        return_stats (:obj:`bool`, optional): Returns dictionary of statistics, default False

    Returns:
        dictionary: optionally returns a dictionary of statistics

    Raises:
        ValueError: Shapes of `y_true` and `y_pred` do not match.

    Warning:
        Comparing labeled to unlabeled data will produce very low accuracy scores.
        Make sure to input the same type of data for `y_true` and `y_pred`
    """

    def _round(x):
        return round(x, ndigits)

    # Apply transformation if requested
    if transform != None:
        y_true = _transform_masks(y_true,transform)

    # Select specified channel
    y_true = y_true[:,:,:,channel_index]
    y_pred = y_pred[:,:,:,channel_index]

    if y_pred.shape != y_true.shape:
        raise ValueError('Shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of y_true after transform is: {}'.format(
                             y_pred.shape, y_true.shape))

    if y_pred.sum() == 0 and y_true.sum() == 0:
        logging.warning('DICE score is technically 1.0, '
                        'but prediction and truth arrays are empty. ')
        return 1.0

    # Threshold to boolean then convert to binary 0,1
    pred = (y_pred >= threshold).astype('int')
    truth = (y_true >= threshold).astype('int')

    # where pred and truth are both nonzero
    intersection = pred * truth  

    # Add to find union and reset to binary
    union = (pred + truth != 0).astype('int')

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
        return({
            'dice':dice,
            'jaccard':jaccard,
            'precision':precision,
            'recall':recall,
            'Fmeasure':Fmeasure
        })
