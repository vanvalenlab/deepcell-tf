"""
callbacks.py

Custom callbacks

@author: cpavelchek, willgraf
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from skimage.measure import label
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.callbacks import Callback

from .image_generators import _transform_masks


class PixelIoU(Callback):
    """At the end of training, calculate the pixel based
    Intersection over Union matrix and Dice/Jaccard scores
    """
    def __init__(self,
                 X,
                 y,
                 dice_iou_threshold=.5,
                 merge_iou_threshold=1e-5,
                 transform=None,
                 transform_kwargs={},
                 crop_size=32,
                 ndigits=4):
        self.X = X
        self.y = _transform_masks(y, transform=transform, **transform_kwargs)
        self.dice_iou_threshold = dice_iou_threshold
        self.merge_iou_threshold = merge_iou_threshold
        self.crop_size = crop_size
        self.ndigits = ndigits
        super(PixelIoU, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        def _round(x):
            return round(x, self.ndigits)

        y_true = self.y
        y_pred = self.model.predict(self.X)
        if isinstance(y_pred, list):
            y_pred = y_pred[-1]

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
        print('Pixel IoU - Dice/F1:', _round(dice), '- Jaccard:', _round(jaccard))


class ObjectIoU(Callback):
    """At the end of training, calculate the object based
    Intersection over Union matrix and Dice/Jaccard scores
    """
    def __init__(self,
                 X,
                 y,
                 dice_iou_threshold=.5,
                 merge_iou_threshold=1e-5,
                 transform=None,
                 transform_kwargs={},
                 crop_size=32,
                 ndigits=4):
        self.X = X
        self.y = _transform_masks(y, transform=transform, **transform_kwargs)
        self.dice_iou_threshold = dice_iou_threshold
        self.merge_iou_threshold = merge_iou_threshold
        self.crop_size = crop_size
        self.ndigits = ndigits
        super(ObjectIoU, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        def _round(x):
            return round(x, self.ndigits)

        y_true = self.y
        y_pred = self.model.predict(self.X)
        if isinstance(y_pred, list):
            y_pred = y_pred[-1]

        stats_iou_matrix = get_iou_matrix_quick(
            y_true, y_pred, self.dice_iou_threshold, self.crop_size)

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
            y_true, y_pred, self.merge_iou_threshold, self.crop_size)

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

        print('Object IoU - dice/F1 index:', _round(dice), 'jaccard index:', _round(jaccard))
        print('cells predicted:', stats_iou_matrix.shape[1], ' - cells true:', stats_iou_matrix.shape[0])
        print('incorrect merges:', merged, ' - ', _round(perc_merged), ' % of truth')
        print('incorrect divisions:', divided, ' - ', _round(perc_divided), ' % of truth')

        print('#false positives: {}\t% of total error: {}\t% of predicted incorrect: {}'.format(
            _round(false_positives), _round(false_pos_perc_err), _round(false_pos_perc_pred)))

        print('#false negatives: {}\t% of total error: {}\t% of ground truth missed: {}'.format(
            _round(false_negatives), _round(false_neg_perc_err), _round(false_neg_perc_truth)))


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
    labeled_truth = label(y_true, connectivity=2)

    iou_matrix = np.zeros((labeled_truth.max(), y_pred.max()))

    # crop input images and calculate the iou's for the cells present
    for x in range(0, im_size, crop_size):
        for y in range(0, im_size, crop_size):
            crop_pred = y_pred[x:x + crop_size, y:y + crop_size]
            crop_truth = labeled_truth[x:x + crop_size, y:y + crop_size]
            iou_matrix = calc_cropped_ious(crop_truth, crop_pred, threshold, iou_matrix)
    return iou_matrix
