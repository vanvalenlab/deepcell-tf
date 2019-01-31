"""
metrics.py

Custom error metrics

@author: cpavelchek, msschwartz21
"""
import datetime
import os
import json

import numpy as np
import skimage.io
import skimage.measure
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
    iou_matrix = np.zeros((y_true.max(), y_pred.max()))

    # Find an intersection mask of all regions of intersection
    mask = np.logical_and(y_true != 0, y_pred != 0)
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

            iou_matrix[tid - 1, pid - 1] = iou

        else:
            intersection = np.logical_and(
                _joint_or(y_true, tid), _joint_or(y_pred, pid))
            union = np.logical_or(_joint_or(y_true, tid),
                                  _joint_or(y_pred, pid))
            iou = np.sum(intersection) / np.sum(union)

            for t in tid:
                for p in pid:
                    iou_matrix[t - 1, p - 1] = iou

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
    # Single hit corresponds to predicted cells that match to true cells
    pred_true_pos = np.count_nonzero(truth_proj == 1)
    # Empty hits indicate predicted cells that do not exist in true cells
    false_pos = np.count_nonzero(truth_proj == 0)
    # More than 2 hits indicates more than 2 true cells corresponding to 1 predicted cell
    merge = np.count_nonzero(truth_proj >= 2)

    # Calc dice jaccard stats for objects
    dice, jaccard = get_dice_jaccard(iou_matrix)

    return {
        'true_cells': true_cells,
        'pred_cells': pred_cells,
        'false_neg': false_neg,
        'split': split,
        'pred_true_pos': pred_true_pos,
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
