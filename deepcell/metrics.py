# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom metrics for pixel-based and object-based classification accuracy.

The schema for this analysis was adopted from the description of object-based
statistics in Caicedo et al. (2018) Evaluation of Deep Learning Strategies for
Nucleus Segmentation in Fluorescence Images. BioRxiv 335216.

The SEG metric was adapted from Maska et al. (2014). A benchmark for comparison
of cell tracking algorithms. Bioinformatics 30, 1609-1617.

The linear classification schema used to match objects in truth and prediction
frames was adapted from Jaqaman et al. (2008). Robust single-particle tracking
in live-cell time-lapse sequences. Nature Methods 5, 695-702.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import datetime
import operator
import math
import decimal

import numpy as np
import pandas as pd
import networkx as nx

from scipy.optimize import linear_sum_assignment

import skimage.io
import skimage.measure
from skimage.segmentation import relabel_sequential
from sklearn.metrics import confusion_matrix

from tensorflow.python.platform import tf_logging as logging


def stats_pixelbased(y_true, y_pred):
    """Calculates pixel-based statistics
    (Dice, Jaccard, Precision, Recall, F-measure)

    Takes in raw prediction and truth data in order to calculate accuracy
    metrics for pixel based classfication. Statistics were chosen according
    to the guidelines presented in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.

    Args:
        y_true (3D np.array): Binary ground truth annotations for a single
            feature, (batch,x,y)
        y_pred (3D np.array): Binary predictions for a single feature,
            (batch,x,y)

    Returns:
        dictionary: Containing a set of calculated statistics

    Raises:
        ValueError: Shapes of `y_true` and `y_pred` do not match.

    Warning:
        Comparing labeled to unlabeled data will produce low accuracy scores.
        Make sure to input the same type of data for `y_true` and `y_pred`
    """

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
    precision = intersection.sum() / pred.sum()
    recall = intersection.sum() / truth.sum()
    Fmeasure = (2 * precision * recall) / (precision + recall)

    return {
        'dice': dice,
        'jaccard': jaccard,
        'precision': precision,
        'recall': recall,
        'Fmeasure': Fmeasure
    }


class ObjectAccuracy(object):
    """Classifies object prediction errors as TP, FP, FN, merge or split

    The schema for this analysis was adopted from the description of
    object-based statistics in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.
    The SEG metric was adapted from Maska et al. (2014). A benchmark for
    comparison of cell tracking algorithms.
    Bioinformatics 30, 1609-1617.
    The linear classification schema used to match objects in truth and
    prediction frames was adapted from Jaqaman et al. (2008).
    Robust single-particle tracking in live-cell time-lapse sequences.
    Nature Methods 5, 695-702.

    Args:
        y_true (2D np.array): Labeled ground truth annotation
        y_pred (2D np.array): Labled object prediction, same size as y_true
        cutoff1 (:obj:`float`, optional): Threshold for overlap in cost matrix,
            smaller values are more conservative, default 0.4
        cutoff2 (:obj:`float`, optional): Threshold for overlap in unassigned
            cells, smaller values are better, default 0.1
        test (:obj:`bool`, optional): Utility variable to control running
            analysis during testing
        seg (:obj:`bool`, optional): Calculates SEG score for cell tracking
            competition

    Raises:
        ValueError: If y_true and y_pred are not the same shape

    Warning:
        Position indicies are not currently collected appropriately

    Todo:
        Implement recording of object indices for each error group
    """

    def __init__(self,
                 y_true,
                 y_pred,
                 cutoff1=0.4,
                 cutoff2=0.1,
                 test=False,
                 seg=False):
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.seg = seg

        if y_pred.shape != y_true.shape:
            raise ValueError('Input shapes must match. Shape of prediction '
                             'is: {}.  Shape of y_true is: {}'.format(
                                 y_pred.shape, y_true.shape))

        # Relabel y_true and y_pred so the labels are consecutive
        y_true, _, _ = relabel_sequential(y_true)
        y_pred, _, _ = relabel_sequential(y_pred)

        self.y_true = y_true
        self.y_pred = y_pred

        self.n_true = len(np.unique(self.y_true)) - 1
        self.n_pred = len(np.unique(self.y_pred)) - 1

        self.n_obj = self.n_true + self.n_pred

        # Initialize error counters
        self.correct_detections = 0
        self.missed_detections = 0
        self.gained_detections = 0

        self.merge = 0
        self.split = 0
        self.catastrophe = 0

        self.gained_det_from_split = 0
        self.missed_det_from_merge = 0
        self.true_det_in_catastrophe = 0
        self.pred_det_in_catastrophe = 0

        # Initialize lists and dicts to store indices where errors occur
        self.correct_indices = {}
        self.correct_indices['y_true'] = []
        self.correct_indices['y_pred'] = []

        self.missed_indices = {}
        self.missed_indices['y_true'] = []

        self.gained_indices = {}
        self.gained_indices['y_pred'] = []

        self.merge_indices = {}
        self.merge_indices['y_true'] = []

        self.split_indices = {}
        self.split_indices['y_true'] = []

        self.catastrophe_indices = {}
        self.catastrophe_indices['y_true'] = []
        self.catastrophe_indices['y_pred'] = []

        # Check if either frame is empty before proceeding
        if self.n_true == 0:
            logging.info('Ground truth frame is empty')
            self.gained_detections += self.n_pred
            self.empty_frame = 'n_true'
        elif self.n_pred == 0:
            logging.info('Prediction frame is empty')
            self.missed_detections += self.n_true
            self.empty_frame = 'n_pred'
        elif test is False:
            self.empty_frame = False
            self._calc_iou()
            self._make_matrix()
            self._linear_assignment()

            # Check if there are loners before proceeding
            if (self.loners_pred.shape[0] == 0) & (self.loners_true.shape[0] == 0):
                pass
            else:
                self._assign_loners()
                self._array_to_graph()
                self._classify_graph()
        else:
            self.empty_frame = False

    def _calc_iou(self):
        """Calculates IoU matrix for each pairwise comparison between true and
        predicted. Additionally, if `seg`==True, records a 1 for each pair of
        objects where $|T\bigcap P| > 0.5 * |T|$
        """

        self.iou = np.zeros((self.n_true, self.n_pred))
        if self.seg is True:
            self.seg_thresh = np.zeros((self.n_true, self.n_pred))

        # Make all pairwise comparisons to calc iou
        for t in range(1, self.y_true.max() + 1):
            for p in range(1, self.y_pred.max() + 1):
                intersection = np.logical_and(self.y_true == t, self.y_pred == p)
                union = np.logical_or(self.y_true == t, self.y_pred == p)
                # Subtract 1 from index to account for skipping 0
                self.iou[t - 1, p - 1] = intersection.sum() / union.sum()
                if (self.seg is True) & \
                   (intersection.sum() > 0.5 * np.sum(self.y_true == t)):
                    self.seg_thresh[t - 1, p - 1] = 1

    def _make_matrix(self):
        """Assembles cost matrix using the iou matrix and cutoff1

        The previously calculated iou matrix is cast into the top left and
        transposed for the bottom right corner. The diagonals of the two
        remaining corners are populated according to `cutoff1`. The lower the
        value of `cutoff1` the more likely it is for the linear sum assignment
        to pick unmatched assignments for objects.
        """

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

    def _linear_assignment(self):
        """Runs linear sun assignment on cost matrix, identifies true positives
        and unassigned true and predicted cells.

        True positives correspond to assignments in the top left or bottom
        right corner. There are two possible unassigned positions: true cell
        unassigned in bottom left or predicted cell unassigned in top right.
        """

        self.results = linear_sum_assignment(self.cm)

        # Map results onto cost matrix
        self.cm_res = np.zeros(self.cm.shape)
        self.cm_res[self.results[0], self.results[1]] = 1

        # Identify direct matches as true positives
        correct_index = np.where(self.cm_res[:self.n_true, :self.n_pred] == 1)
        self.correct_detections += len(correct_index[0])
        self.correct_indices['y_true'].append(correct_index[0])
        self.correct_indices['y_pred'].append(correct_index[1])

        # Calc seg score for true positives if requested
        if self.seg is True:
            iou_mask = self.iou.copy()
            iou_mask[self.seg_thresh == 0] = np.nan
            self.seg_score = np.nanmean(iou_mask[correct_index[0],
                                        correct_index[1]])

        # Collect unassigned cells
        self.loners_pred, _ = np.where(
            self.cm_res[-self.n_pred:, :self.n_pred] == 1)
        self.loners_true, _ = np.where(
            self.cm_res[:self.n_true, -self.n_true:] == 1)

    def _assign_loners(self):
        """Generate an iou matrix for the subset unassigned cells
        """

        self.n_pred2 = len(self.loners_pred)
        self.n_true2 = len(self.loners_true)
        self.n_obj2 = self.n_pred2 + self.n_true2

        self.cost_l = np.zeros((self.n_true2, self.n_pred2))

        for i, t in enumerate(self.loners_true):
            for j, p in enumerate(self.loners_pred):
                self.cost_l[i, j] = self.iou[t, p]

        self.cost_l_bin = self.cost_l > self.cutoff2

    def _array_to_graph(self):
        """Transform matrix for unassigned cells into a graph object

        In order to cast the iou matrix into a graph form, we treat each
        unassigned cell as a node. The iou values for each pair of cells is
        treated as an edge between nodes/cells. Any iou values equal to 0 are
        dropped because they indicate no overlap between cells.
        """

        # Use meshgrid to get true and predicted cell index for each val
        tt, pp = np.meshgrid(self.loners_true, self.loners_pred, indexing='ij')

        df = pd.DataFrame({
            'true': tt.flatten(),
            'pred': pp.flatten(),
            'weight': self.cost_l_bin.flatten()
        })

        # Change cell index to str names
        df['true'] = 'true_' + df['true'].astype('str')
        df['pred'] = 'pred_' + df['pred'].astype('str')

        # Drop 0 weights to only retain overlapping cells
        dfedge = df.drop(df[df['weight'] == 0].index)

        # Create graph from edges
        self.G = nx.from_pandas_edgelist(dfedge, source='true', target='pred')

        # Add nodes to ensure all cells are included
        nodes_true = ['true_' + str(node) for node in self.loners_true]
        nodes_pred = ['pred_' + str(node) for node in self.loners_pred]
        nodes = nodes_true + nodes_pred
        self.G.add_nodes_from(nodes)

    def _classify_graph(self):
        """Assign each node in graph to an error type

        Nodes with a degree (connectivity) of 0 correspond to either false
        positives or false negatives depending on the origin of the node from
        either the predicted objects (false positive) or true objects
        (false negative). Any nodes with a connectivity of 1 are considered to
        be true positives that were missed during linear assignment.
        Finally any nodes with degree >= 2 are indicative of a merge or split
        error. If the top level node is a predicted cell, this indicates a merge
        event. If the top level node is a true cell, this indicates a split event.
        """

        # Find subgraphs, e.g. merge/split

        subgraphs = [self.G.subgraph(c) for c in nx.connected_components(self.G)]
        for g in nx.connected_component_subgraphs(self.G):

            # Get the highest degree node
            k = max(dict(g.degree).items(), key=operator.itemgetter(1))[0]

            # Map index back to original cost matrix
            index = int(k.split('_')[-1])
            # Process degree 0 nodes
            if g.degree[k] == 0:
                if 'pred' in k:
                    self.gained_detections += 1
                    self.gained_indices['y_pred'].append(index)
                if 'true' in k:
                    self.missed_detections += 1
                    self.missed_indices['y_true'].append(index)

            # Process degree 1 nodes
            if g.degree[k] == 1:
                for node in g.nodes:
                    node_index = int(node.split('_')[-1])
                    if 'pred' in node:
                        self.gained_detections += 1
                        self.gained_indices['y_pred'].append(node_index)
                    if 'true' in node:
                        self.missed_detections += 1
                        self.missed_indices['y_true'].append(node_index)

            # Process multi-degree nodes
            elif g.degree[k] > 1:
                node_type = k.split('_')[0]
                nodes = g.nodes()
                # Check whether the subgraph has multiple types of the
                # highest degree node (true or pred)
                n_node_type = np.sum([node_type in node for node in nodes])
                # If there is only one of the high degree node type in the
                # sub graph, then we have either a merge or a split
                if n_node_type == 1:
                    # Check for merges
                    if 'pred' in node_type:
                        self.merge += 1
                        self.missed_det_from_merge += len(nodes) - 2
                        merge_indices = [int(node.split('_')[-1])
                                         for node in nodes if 'true' in node]
                        self.merge_indices['y_true'] += merge_indices
                    # Check for splits
                    elif 'true' in node_type:
                        self.split += 1
                        self.gained_det_from_split += len(nodes) - 2
                        self.split_indices['y_true'].append(index)

                # If there are multiple types of the high degree node,
                # then we have a catastrophe
                else:
                    self.catastrophe += 1
                    true_indices = [int(node.split('_')[-1]) for node in nodes if 'true' in node]
                    pred_indices = [int(node.split('_')[-1]) for node in nodes if 'pred' in node]

                    self.true_det_in_catastrophe = len(true_indices)
                    self.pred_det_in_catastrophe = len(pred_indices)

                    self.catastrophe_indices['y_true'] += true_indices
                    self.catastrophe_indices['y_pred'] += pred_indices

            # Save information about the cells involved in the different error types
            gained_label_image = np.zeros_like(self.y_pred)
            for l in self.gained_indices['y_pred']:
                gained_label_image[self.y_pred == l] = l
            self.gained_props = skimage.measure.regionprops(gained_label_image)

            missed_label_image = np.zeros_like(self.y_true)
            for l in self.missed_indices['y_true']:
                missed_label_image[self.y_true == l] = l
            self.missed_props = skimage.measure.regionprops(missed_label_image)

            merge_label_image = np.zeros_like(self.y_true)
            for l in self.merge_indices['y_true']:
                merge_label_image[self.y_true == l] = l
            self.merge_props = skimage.measure.regionprops(merge_label_image)

            split_label_image = np.zeros_like(self.y_true)
            for l in self.split_indices['y_true']:
                split_label_image[self.y_true == l] = l
            self.split_props = skimage.measure.regionprops(merge_label_image)

    def print_report(self):
        """Print report of error types and frequency
        """

        print(self.save_to_dataframe())

    def save_to_dataframe(self):
        """Save error results to a pandas dataframe

        Returns:
            pd.DataFrame: Single row dataframe with error types as columns
        """
        D = {
            'n_pred': self.n_pred,
            'n_true': self.n_true,
            'correct_detections': self.correct_detections,
            'missed_detections': self.missed_detections,
            'gained_detections': self.gained_detections,
            'missed_det_from_merge': self.missed_det_from_merge,
            'gained_det_from_split': self.gained_det_from_split,
            'true_det_in_catastrophe': self.true_det_in_catastrophe,
            'pred_det_in_catastrophe': self.pred_det_in_catastrophe,
            'merge': self.merge,
            'split': self.split,
            'catastrophe': self.catastrophe
        }

        if self.seg is True:
            D['seg'] = self.seg_score

        # Calculate jaccard index for pixel classification
        pixel_stats = stats_pixelbased(self.y_true != 0, self.y_pred != 0)
        D['jaccard'] = pixel_stats['jaccard']

        df = pd.DataFrame(D, index=[0], dtype='float64')

        # Change appropriate columns to int dtype
        col = ['n_pred', 'n_true', 'correct_detections', 'missed_detections', 'gained_detections',
               'missed_det_from_merge', 'gained_det_from_split', 'true_det_in_catastrophe',
               'pred_det_in_catastrophe', 'merge', 'split', 'catastrophe']
        df[col] = df[col].astype('int')

        return df


def to_precision(x, p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    decimal.getcontext().prec = p
    return decimal.Decimal(x)


class Metrics(object):
    """Class to calculate and save various classification metrics

    Args:
        model_name (str): Name of the model which determines output file names
        outdir (:obj:`str`, optional): Directory to save json file, default ''
        cutoff1 (:obj:`float`, optional): Threshold for overlap in cost matrix,
            smaller values are more conservative, default 0.4
        cutoff2 (:obj:`float`, optional): Threshold for overlap in unassigned
            cells, smaller values are better, default 0.1
        pixel_threshold (:obj:`float`, optional): Threshold for converting
            predictions to binary
        ndigits (:obj:`int`, optional): Sets number of digits for rounding,
            default 4
        feature_key (:obj:`list`, optional): List of strings, feature names
        json_notes (:obj:`str`, optional): Str providing any additional
            information about the model
        seg (:obj:`bool`, optional): Calculates SEG score for
            cell tracking competition

    Examples:
        >>> from deepcell import metrics
        >>> m = metrics.Metrics('model_name')
        >>> m.run_all(
                y_true_lbl,
                y_pred_lbl,
                y_true_unlbl,
                y_true_unlbl)

        >>> m.all_pixel_stats(y_true_unlbl,y_pred_unlbl)
        >>> m.calc_obj_stats(y_true_lbl,y_pred_lbl)
        >>> m.save_to_json(m.output)
    """

    def __init__(self, model_name,
                 outdir='',
                 cutoff1=0.4,
                 cutoff2=0.1,
                 pixel_threshold=0.5,
                 ndigits=4,
                 crop_size=None,
                 return_iou=False,
                 feature_key=[],
                 json_notes='',
                 seg=False):

        self.model_name = model_name
        self.outdir = outdir
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.pixel_threshold = pixel_threshold
        self.ndigits = ndigits
        self.crop_size = crop_size
        self.return_iou = return_iou
        self.feature_key = feature_key
        self.json_notes = json_notes
        self.seg = seg

        # Initialize output list to collect stats
        self.output = []

    def all_pixel_stats(self, y_true, y_pred):
        """Collect pixel statistics for each feature.

        y_true should have the appropriate transform applied to match y_pred.
        Each channel is converted to binary using the threshold
        `pixel_threshold` prior to calculation of accuracy metrics.

        Args:
            y_true (4D np.array): Ground truth annotations after transform
            y_pred (4D np.array): Model predictions without labeling

        Raises:
            ValueError: If y_true and y_pred are not the same shape
        """

        if y_pred.shape != y_true.shape:
            raise ValueError('Input shapes need to match. Shape of prediction '
                             'is: {}.  Shape of y_true is: {}'.format(
                                 y_pred.shape, y_true.shape))

        n_features = y_pred.shape[-1]

        # Intialize df to collect pixel stats
        self.pixel_df = pd.DataFrame()

        # Set numeric feature key if existing key is not write length
        if n_features != len(self.feature_key):
            self.feature_key = range(n_features)

        for i, k in enumerate(self.feature_key):
            yt = y_true[:, :, :, i] > self.pixel_threshold
            yp = y_pred[:, :, :, i] > self.pixel_threshold
            stats = stats_pixelbased(yt, yp)
            self.pixel_df = self.pixel_df.append(
                pd.DataFrame(stats, index=[k]))

        # Save stats to output dictionary
        self.output = self.output + self.pixel_df_to_dict(self.pixel_df)

        # Calculate confusion matrix
        self.cm = self.calc_pixel_confusion_matrix(y_true, y_pred)
        self.output.append(dict(
            name='confusion_matrix',
            value=self.cm.tolist(),
            feature='all',
            stat_type='pixel'
        ))

        self.print_pixel_report()

    def pixel_df_to_dict(self, df):
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

    def calc_pixel_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix for pixel classification data.

        Args:
            y_true (4D np.array): Ground truth annotations after any
                necessary transformations
            y_pred (4D np.array): Prediction array

        Returns:
            confusion_matrix: nxn array determined by number of features
        """

        # Argmax collapses on feature dimension to assign class to each pixel
        # Flatten is requiremed for confusion matrix
        y_true = y_true.argmax(axis=-1).flatten()
        y_pred = y_pred.argmax(axis=-1).flatten()

        return confusion_matrix(y_true, y_pred)

    def print_pixel_report(self):
        """Print report of pixel based statistics
        """

        print('\n____________Pixel-based statistics____________\n')
        print(self.pixel_df)
        print('\nConfusion Matrix')
        print(self.cm)

    def calc_object_stats(self, y_true, y_pred):
        """Calculate object statistics and save to output

        Loops over each frame in the zeroth dimension, which should pass in
        a series of 2D arrays for analysis. `metrics.split_stack` can be
        used to appropriately reshape the input array if necessary

        Args:
            y_true (3D np.array): Labeled ground truth annotations
            y_pred (3D np.array): Labeled prediction mask
        """
        self.stats = pd.DataFrame()

        for i in range(y_true.shape[0]):
            o = ObjectAccuracy(y_true[i],
                               y_pred[i],
                               cutoff1=self.cutoff1,
                               cutoff2=self.cutoff2,
                               seg=self.seg)
            self.stats = self.stats.append(o.save_to_dataframe())
            if i % 500 == 0:
                logging.info('{} samples processed'.format(i))

        # Write out summed statistics
        for k, v in self.stats.iteritems():
            if k == 'seg':
                self.output.append(dict(
                    name=k,
                    value=v.mean(),
                    feature='mean',
                    stat_type='object'
                ))
            else:
                self.output.append(dict(
                    name=k,
                    value=v.sum().astype('float64'),
                    feature='sum',
                    stat_type='object'
                ))

        self.print_object_report()

    def print_object_report(self):
        """Print neat report of object based statistics
        """

        print('\n____________Object-based statistics____________\n')
        print('Number of true cells:\t\t', self.stats['n_true'].sum())
        print('Number of predicted cells:\t', self.stats['n_pred'].sum())

        print('\nCorrect detections:  {}\tRecall: {}%'.format(
            int(self.stats['correct_detections'].sum()),
            to_precision(100 * self.stats['correct_detections'].sum() / self.stats['n_true'].sum(),
                         self.ndigits)))
        print('Incorrect detections: {}\tPrecision: {}%'.format(
            int(self.stats['n_pred'].sum() - self.stats['correct_detections'].sum()),
            to_precision(100 * self.stats['correct_detections'].sum() / self.stats['n_pred'].sum(),
                         self.ndigits)))

        total_err = (self.stats['gained_detections'].sum()
                     + self.stats['missed_detections'].sum()
                     + self.stats['split'].sum()
                     + self.stats['merge'].sum()
                     + self.stats['catastrophe'].sum())

        print('\nGained detections: {}\tPerc Error: {}%'.format(
              int(self.stats['gained_detections'].sum()),
              to_precision(100 * self.stats['gained_detections'].sum() / total_err, self.ndigits)))
        print('Missed detections: {}\tPerc Error: {}%'.format(
              int(self.stats['missed_detections'].sum()),
              to_precision(100 * self.stats['missed_detections'].sum() / total_err, self.ndigits)))
        print('Merges: {}\t\tPerc Error: {}%'.format(
              int(self.stats['merge'].sum()),
              to_precision(100 * self.stats['merge'].sum() / total_err, self.ndigits)))
        print('Splits: {}\t\tPerc Error: {}%'.format(
              int(self.stats['split'].sum()),
              to_precision(100 * self.stats['split'].sum() / total_err, self.ndigits)))
        print('Catastrophes: {}\t\tPerc Error: {}%\n'.format(
              int(self.stats['catastrophe'].sum()),
              to_precision(100 * self.stats['catastrophe'].sum() / total_err, self.ndigits)))

        print('Gained detections from splits: {}'.format(
              int(self.stats['gained_det_from_split'].sum())))
        print('Missed detections from merges: {}'.format(
              int(self.stats['missed_det_from_merge'].sum())))
        print('True detections involved in catastrophes: {}'.format(
              int(self.stats['true_det_in_catastrophe'].sum())))
        print('Predicted detections involved in catastrophes: {}'.format(
              int(self.stats['pred_det_in_catastrophe'].sum())), '\n')

        if self.seg is True:
            print('SEG:', to_precision(self.stats['seg'].mean(), self.ndigits), '\n')

        print('Average Pixel IOU (Jaccard Index):',
              to_precision(self.stats['jaccard'].mean(), self.ndigits), '\n')

    def run_all(self,
                y_true_lbl,
                y_pred_lbl,
                y_true_unlbl,
                y_pred_unlbl):
        """Runs pixel and object base statistics and ouputs to file

        Args:
            y_true_lbl (3D np.array): Labeled ground truth annotation,
                (sample, x, y)
            y_pred_lbl (3D np.array): Labeled prediction mask,
                (sample, x, y)
            y_true_unlbl (4D np.array): Ground truth annotation after necessary
                transforms, (sample, x, y, feature)
            y_pred_unlbl (4D np.array): Predictions, (sample, x, y, feature)
        """

        logging.info('Starting pixel based statistics')
        self.all_pixel_stats(y_true_unlbl, y_pred_unlbl)

        logging.info('Starting object based statistics')
        self.calc_object_stats(y_true_lbl, y_pred_lbl)

        self.save_to_json(self.output)

    def save_to_json(self, L):
        """Save list of dictionaries to json file with file metadata

        Args:
            L (list): List of metric dictionaries
        """

        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        outname = os.path.join(
            self.outdir, self.model_name + '_' + todays_date + '.json')

        # Configure final output
        D = {}

        # Record metadata
        D['metadata'] = dict(
            model_name=self.model_name,
            date=todays_date,
            notes=self.json_notes
        )

        # Record metrics
        D['metrics'] = L

        with open(outname, 'w') as outfile:
            json.dump(D, outfile)

        logging.info('Saved to {}'.format(outname))


def split_stack(arr, batch, n_split1, axis1, n_split2, axis2):
    """Crops an array in the width and height dimensions to produce
    a stack of smaller arrays

    Args:
        arr (np.array): Array to be split with at least 2 dimensions
        batch (bool): True if the zeroth dimension of arr is a batch or
            frame dimension
        n_split1 (int): Number of sections to produce from the first split axis
            Must be able to divide arr.shape[axis1] evenly by n_split1
        axis1 (int): Axis on which to perform first split
        n_split2 (int): Number of sections to produce from the second split axis
            Must be able to divide arr.shape[axis2] evenly by n_split2
        axis2 (int): Axis on which to perform first split

    Returns:
        np.array: Array after dual splitting with frames in the zeroth dimension

    Raises:
        ValueError: arr.shape[axis] must be evenly divisible by n_split
            for both the first and second split

    Examples:
        >>> from deepcell import metrics
        >>> from numpy import np
        >>> arr = np.ones((10, 100, 100, 1))
        >>> out = metrics.test_split_stack(arr, True, 10, 1, 10, 2)
        >>> out.shape
        (1000, 10, 10, 1)
        >>> arr = np.ones((100, 100, 1))
        >>> out = metrics.test_split_stack(arr, False, 10, 1, 10, 2)
        >>> out.shape
        (100, 10, 10, 1)
    """
    # Check that n_split will divide equally
    if ((arr.shape[axis1] % n_split1) != 0) | ((arr.shape[axis2] % n_split2) != 0):
        raise ValueError(
            'arr.shape[axis] must be evenly divisible by n_split'
            'for both the first and second split')

    split1 = np.split(arr, n_split1, axis=axis1)

    # If batch dimension doesn't exist, create and adjust axis2
    if batch is False:
        split1con = np.stack(split1)
        axis2 += 1
    else:
        split1con = np.concatenate(split1, axis=0)

    split2 = np.split(split1con, n_split2, axis=axis2)
    split2con = np.concatenate(split2, axis=0)

    return split2con
