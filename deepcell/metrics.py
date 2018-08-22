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
        return prediction, mask

    # otherwise, pad prediction to imsize and zero out the outer layer of mask
    mask = mask[win_size:-win_size, win_size:-win_size]
    mask = np.pad(mask, (win_size, win_size), 'constant')
    prediction = np.pad(prediction, (win_size, win_size), 'constant')
    return prediction, mask


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
    for x in range(im_size, crop_size):
        for y in range(im_size, crop_size):
            crop_pred = y_pred[x:x + crop_size, y:y + crop_size]
            crop_truth = y_true[x:x + crop_size, y:y + crop_size]
            iou_matrix = calc_cropped_ious(crop_pred, crop_truth, threshold, iou_matrix)
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


def count_false_negatives(iou_matrix):
    """Compare each predicted mask with ground truth and count the
    false negatives (failed to predict a ground truth mask)
    """
    false_negatives = 0
    # for each ground truth cellID
    for n in range(iou_matrix.shape[0]):
        # check each predicted mask
        for m in range(iou_matrix.shape[1]):
            # if any of the mask predictions match the cellID
            # move on to the next cell
            if iou_matrix[n, m] == 1:
                break
        else:
            # if no matches are found, then a false negative has occurred.
            false_negatives += 1
    return false_negatives


def count_false_positives(iou_matrix):
    """Compare each predicted mask with ground truth and count the
    false positives (predicted a mask where there is no ground truth mask)
    """
    false_positives = 0
    # for each mask prediction
    for m in range(iou_matrix.shape[1]):
        # check all ground truth cells
        for n in range(iou_matrix.shape[0]):
            # if any of the ground truth cells match the predicted mask
            # move on to the next
            if iou_matrix[n, m] == 1:
                break
        else:
            # if no matches are found, then a false positive has occured
            false_positives += 1
    return false_positives


def count_false_divisions(iou_matrix):
    """Count the number of incorrectly divided cells.
    For each ground truth mask, find the number of predicted masks.
    Each predicted mask > 1 counts as a "false division"
    """
    # for each unique cell in the ground truth mask
    # count the number of overlapping predicted masks.
    # every predicted mask beyond the first represents an incorrect division.
    divided = 0
    for n in range(iou_matrix.shape[0]):
        counter = 0
        for m in range(iou_matrix.shape[1]):
            if iou_matrix[n, m] == 1:
                counter += 1
        # 1 overlap is expected, but extras are false "divisions"
        if counter > 1:
            divided += counter - 1
    return divided


def count_false_merges(iou_matrix):
    """Count the number of incorrectly merged cells.
    For each predicted mask, find the number of overlaps with the ground truth.
    Each ground truth mask > 1 counts as a "false merge"
    """
    # for each predicted mask, count the # of overlapping cells in the ground truth.
    # every overlapping cell beyond the first represents an incorrect merge
    merged = 0
    for m in range(iou_matrix.shape[1]):
        counter = 0
        for n in range(1, iou_matrix.shape[0]):
            if iou_matrix[n, m] == 1:
                counter += 1
        # 1 overlap is expected, but extras are false "merges"
        if counter > 1:
            merged += counter - 1
    return merged


def stats_objectbased(y_true,
                      y_pred,
                      dice_iou_threshold=.5,
                      merge_iou_threshold=1e-5,
                      ndigits=4,
                      crop_size=32):
    """
    Calculate summary statistics (DICE/Jaccard index and confusion matrix)
    on a per-object basis
    """
    stats_iou_matrix = get_iou_matrix_quick(
        y_pred, y_true, dice_iou_threshold, crop_size)

    dice, jaccard = get_dice_jaccard(stats_iou_matrix)

    # Calculate false negative/positive rates
    false_neg = count_false_negatives(stats_iou_matrix)
    false_pos = count_false_positives(stats_iou_matrix)

    false_pos_perc_err = false_pos / (false_pos + false_neg)
    false_neg_perc_err = false_neg / (false_pos + false_neg)

    false_pos_perc_pred = false_pos / stats_iou_matrix.shape[1]
    false_neg_perc_truth = false_neg / stats_iou_matrix.shape[0]

    # Calculate merge/division error rates
    merge_div_iou_matrix = get_iou_matrix_quick(
        y_pred, y_true, merge_iou_threshold, crop_size)

    divided = count_false_divisions(merge_div_iou_matrix)
    merged = count_false_merges(merge_div_iou_matrix)

    perc_merged = merged / stats_iou_matrix.shape[0]
    perc_divided = divided / stats_iou_matrix.shape[0]

    print('\n____________________Object-based statistics____________________\n')
    print('Intersection over Union thresholded at:', dice_iou_threshold)
    print('dice/F1 index: {}\njaccard index:{}\n'.format(
        round(dice, ndigits), round(jaccard, ndigits)))

    print('#false positives: {}\t%% of total error: {}\t%% of predicted incorrect: {}'.format(
        round(false_pos, ndigits),
        round(false_pos_perc_err, ndigits),
        round(false_pos_perc_pred, ndigits)))

    print('#false negatives: {}\t%% of total error: {}\t%% of ground truth missed: {}'.format(
        round(false_neg, ndigits),
        round(false_neg_perc_err, ndigits),
        round(false_neg_perc_truth, ndigits)))

    print('\nIntersection over Union thresholded at:', merge_iou_threshold)
    print('#incorrect merges: {}\t%% of ground truth merged: {}'.format(
        merged, round(perc_merged, ndigits)))
    print('#incorrect divisions: {}\t%% of ground truth divided: {}'.format(
        divided, round(perc_divided, ndigits)))


def stats_pixelbased(y_true, y_pred, ndigits=4):
    """Calculates pixel-based dice and jaccard scores, and prints them"""
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

    return dice, jaccard
