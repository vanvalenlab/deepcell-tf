"""
metrics.py

Custom error metrics

@author: cpavelchek
"""

import warnings
import numpy as np
import skimage.io
import skimage.measure


def im_prep(prediction, mask, win_size):
    """Reads images into ndarrays, and trims them to fit each other"""
    # if trimming not needed, return
    if win_size == 0 or prediction.shape == mask.shape:
        return prediction, mask

    # otherwise, pad prediction to imsize and zero out the outer layer of mask
    mask = mask[win_size:-win_size, win_size:-win_size]
    mask = np.pad(mask, (win_size, win_size), 'constant')
    prediction = np.pad(prediction, (win_size, win_size), 'constant')
    return prediction, mask


def calc_cropped_ious(crop_pred, crop_truth, threshold, iou_matrix):
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


def get_iou_matrix_quick(pred, truth, threshold, crop_size, im_size=2048):
    # label ground truth masks, neccesary if not already tagged with cellID numbers
    truth = skimage.measure.label(truth, connectivity=2)

    # create empty intersection over union matrix, with shape n(truth) by m(prediction)
    iou_matrix = np.zeros((truth.max(), pred.max()))

    # crop input images and calculate the iou's for the cells present
    for x in range(im_size, crop_size):
        for y in range(im_size, crop_size):
            crop_pred = pred[x:x + crop_size, y:y + crop_size]
            crop_truth = truth[x:x + crop_size, y:y + crop_size]
            iou_matrix = calc_cropped_ious(crop_pred, crop_truth, threshold, iou_matrix)
    return iou_matrix


def dice_jaccard_object(pred, truth, threshold=.5, crop_size=256):
    iou_matrix = get_iou_matrix_quick(pred, truth, threshold, crop_size=crop_size)
    iou_sum = np.sum(iou_matrix)
    pred_max = iou_matrix.shape[1] - 1
    truth_max = iou_matrix.shape[0] - 1

    dice_object = 2 * iou_sum / (2 * iou_sum + pred_max - iou_sum + truth_max - iou_sum)
    jaccard_object = dice_object / (2 - dice_object)

    return iou_matrix, dice_object, jaccard_object


def count_false_pos_neg(iou_matrix):

    # Count the number of cellID's in the ground truth mask without a corresponding prediction
    false_neg = 0

    # for each ground truth cellID
    for n in range(iou_matrix.shape[0]):
        counter = 0

        # check all masks
        for m in range(iou_matrix.shape[1]):

            # if any of the mask predictions match the cellID, move on to the next cell
            if iou_matrix[n, m] == 1:
                counter += 1

        # Otherwise, if no matches are found, then a false negative has occurred.
        if counter == 0:
            false_neg += 1

    # Count the number of predicted masks without a corresponding ground-truth cell
    false_pos = 0

    # for each predicted cell
    for m in range(iou_matrix.shape[1]):
        counter = 0

        # check all ground truth cells
        for n in range(iou_matrix.shape[0]):

            # if any of the ground truth cells match the predicted mask, move on to the next
            if iou_matrix[n, m] == 1:
                counter += 1
                continue

        # Otherwise, if no matches are found, then a false positive has occured
        if counter == 0:
            false_pos += 1

    return false_pos, false_neg


def count_merg_div(iou_matrix):
    """Count incorrect merges and incorrect divisons using the IOU matrix"""
    # for each unique cell in the ground truth mask
    # count the number of overlapping predicted masks.
    # every predicted mask beyond the first represents an incorrect division.
    divided = 0
    for n in range(iou_matrix.shape[0]):
        counter = 0
        for m in range(iou_matrix.shape[1]):
            if iou_matrix[n, m] == 1:
                counter += 1
        if counter > 1:
            divided += counter - 1

    # for each predicted mask, count the # of overlapping cells in the ground truth.
    # every overlapping cell beyond the first represents an incorrect merge
    merged = 0
    for m in range(iou_matrix.shape[1]):
        counter = 0
        for n in range(1, iou_matrix.shape[0]):
            if iou_matrix[n, m] == 1:
                counter += 1
        if counter > 1:
            merged += counter - 1

    return merged, divided


def stats_objectbased(pred_input,
                      truth_input,
                      dice_iou_threshold=.5,
                      merge_iou_threshold=1e-5,
                      ndigits=4,
                      crop_size=32):

    # copy inputs so original arrays are not modified
    wshed_pred = np.copy(pred_input)
    wshed_truth = np.copy(truth_input)

    stats_iou_matrix, dice_object, jaccard_object = dice_jaccard_object(
        wshed_pred,
        wshed_truth,
        threshold=dice_iou_threshold,
        crop_size=crop_size)

    false_pos, false_neg = count_false_pos_neg(stats_iou_matrix)

    false_pos_perc_err = false_pos / (false_pos + false_neg)
    false_neg_perc_err = false_neg / (false_pos + false_neg)

    false_pos_perc_pred = false_pos / stats_iou_matrix.shape[1]
    false_neg_perc_truth = false_neg / stats_iou_matrix.shape[0]

    merge_div_iou_matrix = get_iou_matrix_quick(
        wshed_pred,
        wshed_truth,
        threshold=merge_iou_threshold,
        crop_size=crop_size)

    merged, divided = count_merg_div(merge_div_iou_matrix)

    perc_merged = merged / stats_iou_matrix.shape[0]
    perc_divided = divided / stats_iou_matrix.shape[0]

    # round all print percentages to a given limit
    dice_object = round(dice_object, ndigits)
    jaccard_object = round(jaccard_object, ndigits)
    false_pos = round(false_pos, ndigits)
    false_neg = round(false_neg, ndigits)
    false_pos_perc_err = round(false_pos_perc_err, ndigits)
    false_neg_perc_err = round(false_neg_perc_err, ndigits)
    false_pos_perc_pred = round(false_pos_perc_pred, ndigits)
    false_neg_perc_truth = round(false_neg_perc_truth, ndigits)
    perc_merged = round(perc_merged, ndigits)
    perc_divided = round(perc_divided, ndigits)

    print('\n____________________Object-based statistics____________________\n')
    print('Intersection over Union thresholded at:', dice_iou_threshold)
    print('dice/F1 index:', dice_object)
    print('jaccard index:', jaccard_object)

    print('#false positives: {}', false_pos, '  %% of total error:', false_pos_perc_err, '  %% of predicted incorrect:', false_pos_perc_pred)
    print('#false negatives:', false_neg, '  %% of total error:', false_neg_perc_err, '  %% of ground truth missed:', false_neg_perc_truth)

    print('')
    print('Intersection over Union thresholded at:', merge_iou_threshold)
    print('#incorrect merges:', merged, '     %% of ground truth merged:', perc_merged)
    print('#incorrect divisions:', divided, '  %% of ground truth divided:', perc_divided)
    print('')


def stats_pixelbased(pred_input, truth_input, ndigits=4):
    """Calculates pixel-based dice and jaccard scores, and prints them"""
    pred = np.copy(pred_input)
    truth = np.copy(truth_input)

    pred[pred != 0] = 1
    truth[truth != 0] = 1

    if pred.shape != truth.shape:
        raise ValueError('shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of mask is: {}'.format(
                             pred.shape, truth.shape))

    if pred.sum() == 0 and truth.sum() == 0:
        warnings.warn('DICE score is technically 1.0, '
                      'but prediction and truth arrays are empty. ')
        return 1.0

    intersection = np.logical_and(pred, truth)

    dice_pixel = (2 * intersection.sum() / (pred.sum() + truth.sum()))
    jaccard_pixel = dice_pixel / (2 - dice_pixel)

    dice_pixel = round(dice_pixel, ndigits)
    jaccard_pixel = round(jaccard_pixel, ndigits)

    print('\n____________________Pixel-based statistics____________________\n')
    print('dice/F1 index:', dice_pixel)
    print('jaccard index:', jaccard_pixel)
    print('')
    print('')
