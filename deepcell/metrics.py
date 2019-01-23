"""
metrics.py

Custom error metrics

@author: cpavelchek
"""

import numpy as np
import skimage.io
import skimage.measure
from tensorflow.python.platform import tf_logging as logging


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
    Calculate all Intersection over Union values within the cropped input.
    If values are > a threshold, mark them as a hit.
    """
    # for each unique cellID in the given mask...
    for n in np.unique(crop_truth):
        if n == 0:
            continue  # excluding background

        # for each unique cellID in the given prediction...
        for m in np.unique(crop_pred):
            if m == 0:
                continue  # excluding background

            # calculate the intersection over union for
            intersection = np.logical_and(crop_pred == m, crop_truth == n)
            union = np.logical_or(crop_pred == m, crop_truth == n)
            iou = np.sum(intersection) / np.sum(union)

            if iou > threshold:
                iou_matrix[n - 1][m - 1] = 1

    return iou_matrix


def get_iou_matrix_quick(y_true, y_pred, threshold, crop_size, im_size=2048):
    """Calculate Intersection-Over-Union Matrix for ground truth and predictions
    # Arguments
        pred: predicted masks
        truth: ground truth masks
        threshold: If IOU is above threshold, cells are considered overlapping
        crop_size: Cropping images is faster to calculate but less accurate
        im_size: original image size.  (Assumes square images).
    # Returns
        iou_matrix
    """
    # label ground truth masks, neccesary if not already tagged with cellID numbers
    labeled_truth = skimage.measure.label(y_true, connectivity=2)

    iou_matrix = np.zeros((labeled_truth.max(), y_pred.max()))

    # crop input images and calculate the iou's for the cells present
    for x in range(0, im_size, crop_size):
        for y in range(0, im_size, crop_size):
            crop_pred = y_pred[x:x + crop_size, y:y + crop_size]
            crop_truth = labeled_truth[x:x + crop_size, y:y + crop_size]
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


def stats_objectbased(y_true,
                      y_pred,
                      dice_iou_threshold=.5,
                      merge_iou_threshold=1e-5,
                      ndigits=4,
                      crop_size=32):
    """
    Calculate summary statistics (DICE/Jaccard index and confusion matrix)
    for a single channel on a per-object basis

    Relies on  `skimage.measure.label` to define cell objects which are used to calculate stats

    Args:
        y_true (3D np.array): Ground truth annotations for a single channel
        y_pred (3D np.array): Predictions for a single channel
        dice_iou_threshold (:obj:`float`, optional): default, 0.5
        merge_iou_threshold (:obj:`float`, optional): default, 1e-5
        ndigits (:obj:`int`, optional): Sets number of digits for rounding, default 4
        crop_size (:obj:`int`, optional): default 32

    Warning:
        This function currently only accepts single channel data either in a 2D (x,y) 
        or 3D (batch,x,y) form.
        `y_true` and `y_shape` must have the same dimensions

    Raises:
        ValueError: If y_true and y_pred are not the same shape
    """

    def _round(x):
        return round(x,ndigits)

    if y_pred.shape != y_true.shape:
        raise ValueError('Shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of mask is: {}'.format(
                             y_pred.shape, y_true.shape))

    # Convert y input to labeled data
    y_true = skimage.measure.label(y_true, connectivity=2)
    y_pred = skimage.measure.label(y_pred, connectivity=2)

    stats_iou_matrix = get_iou_matrix_quick(
        y_true, y_pred, dice_iou_threshold, crop_size)

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


def stats_pixelbased(y_true, y_pred, ndigits=4):
    """Calculates pixel-based dice and jaccard scores, and prints them

    Args:
        y_true (np.array): Ground truth data
        y_pred (np.array): Predictions
        ndigits (:obj:`int`, optional): Sets number of digits for rounding, default 4

    Returns:
        tuple: dice score, jaccard score

    Warning:
        Currently, will accept various types in input data, e.g. labeled and unlabeled.
        Comparing labeled to unlabeled data will produce very low accuracy scores.
        Make sure to input the same type of data for `y_true` and `y_pred`

    """
    if y_pred.shape != y_true.shape:
        raise ValueError('Shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of mask is: {}'.format(
                             y_pred.shape, y_true.shape))

    if y_pred.sum() == 0 and y_true.sum() == 0:
        logging.warning('DICE score is technically 1.0, '
                        'but prediction and truth arrays are empty. ')
        return 1.0

    pred = (y_pred != 0).astype('int')
    truth = (y_true != 0).astype('int')

    intersection = pred * truth  # where pred and truth are both nonzero

    dice = (2 * intersection.sum() / (pred.sum() + truth.sum()))
    jaccard = dice / (2 - dice)

    print('\n____________________Pixel-based statistics____________________\n')
    print('dice/F1 index: {}\njaccard index: {}'.format(
        round(dice, ndigits), round(jaccard, ndigits)))

    acc = np.count_nonzero(np.logical_and(pred, truth)) / np.count_nonzero(truth)
    print('Accuracy: {}%'.format(100 * round(acc, ndigits)))

    return dice, jaccard
