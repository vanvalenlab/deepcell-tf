# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""Anchor Box generator script that adapts keras_retinanet.utils.anchors 
to include masks during training. """

import numpy as np

from keras_retinanet.utils.compute_overlap import compute_overlap


def anchor_targets_bbox(anchors,
                        annotations,
                        num_classes,
                        mask_shape=None,
                        negative_overlap=0.4,
                        positive_overlap=0.5,
                        **kwargs):
    """Generate anchor targets for bbox detection.

    Args:
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used
            to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors
            (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors
            (all anchors with overlap > positive_overlap are positive).

    Returns:
        labels: np.array of shape (A, num_classes) where a row consists of 0
            for negative and 1 for positive for a certain class.
        annotations: np.array of shape (A, 5) for (x1, y1, x2, y2, label)
            containing the annotations corresponding to each anchor or 0 if
            there is no corresponding anchor.
        anchor_states: np.array of shape (N,) containing the state of an anchor
            (-1 for ignore, 0 for bg, 1 for fg).
    """
    # anchor states: 1 is positive, 0 is negative, -1 is dont care
    anchor_states = np.zeros((anchors.shape[0],))
    labels = np.zeros((anchors.shape[0], num_classes))

    if annotations.shape[0]:
        # obtain indices of gt annotations with the greatest overlap
        overlaps = compute_overlap(anchors.astype(np.float64),
                                   annotations.astype(np.float64))
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # assign "dont care" labels
        positive_indices = max_overlaps >= positive_overlap
        ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
        anchor_states[ignore_indices] = -1
        anchor_states[positive_indices] = 1

        # compute box regression targets
        annotations = annotations[argmax_overlaps_inds]

        # compute target class labels
        labels[positive_indices, annotations[positive_indices, 4].astype(int)] = 1
    else:
        # no annotations? then everything is background
        annotations = np.zeros((anchors.shape[0], annotations.shape[1]))

    # ignore annotations outside of image
    if mask_shape:
        anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2,
                                     (anchors[:, 1] + anchors[:, 3]) / 2]).T
        indices = np.logical_or(anchors_centers[:, 0] >= mask_shape[1],
                                anchors_centers[:, 1] >= mask_shape[0])
        anchor_states[indices] = -1

    return labels, annotations, anchor_states
