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
"""Utilities plotting data"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tensorflow.python.keras import backend as K


def get_js_video(images, batch=0, channel=0, cmap='jet'):
    """Create a JavaScript video as HTML for visualizing 3D data as a movie"""
    fig = plt.figure()

    ims = []
    for i in range(images.shape[1]):
        im = plt.imshow(images[batch, i, :, :, channel], animated=True, cmap=cmap)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=150, repeat_delay=1000)
    plt.close()
    return ani.to_jshtml()


def draw_box(image, box, color, thickness=2):
    """Draws a box on an image with a given color.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        color: The color of the box.
        thickness: The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """Draws a caption above the box in an image.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_mask(image,
              box,
              mask,
              color=[31, 0, 255],
              binarize_threshold=0.5):
    """Draws a mask in a given box.

    Args:
        image: Three dimensional image to draw on.
        box: Vector of at least 4 values (x1, y1, x2, y2)
            representing a box in the image.
        mask: A 2D float mask which will be reshaped to the size of the box,
            binarized and drawn over the image.
        color: Color to draw the mask with. If the box has 5 values,
            the last value is assumed to be the label and used to
            construct a default color.
        binarize_threshold: Threshold used for binarizing the mask.
    """
    # resize to fit the box
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

    # binarize the mask
    mask = (mask > binarize_threshold).astype('uint8')

    # draw the mask in the image
    mask_image = np.zeros((image.shape[0], image.shape[1]), 'uint8')
    mask_image[box[1]:box[3], box[0]:box[2]] = mask
    mask = mask_image

    # compute a nice border around the mask
    border = mask - cv2.erode(mask, np.ones((5, 5), 'uint8'), iterations=1)

    # apply color to the mask and border
    mask = (np.stack([mask] * 3, axis=2) * color).astype('uint8')
    border = (np.stack([border] * 3, axis=2) * (255, 255, 255)).astype('uint8')

    # draw the mask
    indices = np.where(mask != [0, 0, 0])
    _mask = 0.5 * image[indices[0], indices[1], :] + \
        0.5 * mask[indices[0], indices[1], :]
    image[indices[0], indices[1], :] = _mask

    # draw the border
    indices = np.where(border != [0, 0, 0])
    _border = 0.2 * image[indices[0], indices[1], :] + \
        0.8 * border[indices[0], indices[1], :]
    image[indices[0], indices[1], :] = _border


def draw_masks(image, boxes, masks, color=[31, 0, 255], binarize_threshold=0.5):
    """Draws a list of masks given a list of boxes.

    Args:
        image: Three dimensional image to draw on.
        boxes: Matrix of shape (N, >=4) (at least 4 values: (x1, y1, x2, y2))
            representing boxes in the image.
        masks: Matrix of shape (N, H, W) of N masks of shape (H, W) which will
            be reshaped to the size of the corresponding box, binarized and
            drawn over the image.
        color: Color or to draw the masks with.
        binarize_threshold: Threshold used for binarizing the masks.
    """
    for box, mask in zip(boxes, masks):
        if not any(b == -1 for b in box):
            draw_mask(image, box, mask, color=color,
                      binarize_threshold=binarize_threshold)


def draw_detections(image,
                    boxes,
                    scores,
                    labels,
                    color=[31, 0, 255],
                    label_to_name=None,
                    score_threshold=0.5):
    """Draws detections in an image.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image: The image to draw on.
        boxes: A [N, 4] matrix (x1, y1, x2, y2).
        scores: A list of N classification scores.
        labels: A list of N labels.
        color: The color of the boxes.
        label_to_name: (optional) Functor for mapping a label to a name.
        score_threshold: Threshold used for determining the detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        draw_box(image, boxes[i, :], color=color)

        # draw labels
        name = label_to_name(labels[i]) if label_to_name else labels[i]
        caption = '{0}: {1:.2f}'.format(name, scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image,
                     annotations,
                     color=[31, 0, 255],
                     label_to_name=None):
    """Draws annotations in an image.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image: The image to draw on.
        annotations: A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary
            containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color: The color of the boxes.
        label_to_name: (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert 'bboxes' in annotations
    assert 'labels' in annotations
    assert annotations['bboxes'].shape[0] == annotations['labels'].shape[0]

    for i in range(annotations['bboxes'].shape[0]):
        label = annotations['labels'][i]
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=color)


def cf(x_coord, y_coord, sample_image):
    numrows, numcols = sample_image.shape
    col = int(x_coord + 0.5)
    row = int(y_coord + 0.5)
    if 0 <= col < numcols and 0 <= row < numrows:
        z_coord = sample_image[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x_coord, y_coord, z_coord)
    return 'x=%1.4f, y=%1.4f' % (x_coord, y_coord)


def plot_training_data_2d(X, y, max_plotted=5):
    data_format = K.image_data_format()
    if max_plotted > y.shape[0]:
        max_plotted = y.shape[0]

    label_axis = 1 if K.image_data_format() == 'channels_first' else -1

    fig, ax = plt.subplots(max_plotted, y.shape[label_axis] + 1, squeeze=False)

    for i in range(max_plotted):
        X_i = X[i, 0, :, :] if data_format == 'channels_first' else X[i, :, :, 0]
        ax[i, 0].imshow(X_i, cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x_coord, y_coord):
            return cf(x_coord, y_coord, X_i)

        ax[i, 0].format_coord = form_coord
        ax[i, 0].axes.get_xaxis().set_visible(False)
        ax[i, 0].axes.get_yaxis().set_visible(False)

        for j in range(1, y.shape[label_axis] + 1):
            y_k = y[i, j - 1, :, :] if data_format == 'channels_first' else y[i, :, :, j - 1]
            ax[i, j].imshow(y_k, cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[i, j].axes.get_xaxis().set_visible(False)
            ax[i, j].axes.get_yaxis().set_visible(False)

    plt.show()


def plot_training_data_3d(X, y, num_image_stacks, frames_to_display=5):
    data_format = K.image_data_format()
    fig, ax = plt.subplots(num_image_stacks, frames_to_display + 1, squeeze=False)

    for i in range(num_image_stacks):
        X_i = X[i, 0, :, :] if data_format == 'channels_first' else X[i, :, :, 0]
        ax[i, 0].imshow(X_i, cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x_coord, y_coord):
            return cf(x_coord, y_coord, X_i)

        ax[i, 0].format_coord = form_coord
        ax[i, 0].axes.get_xaxis().set_visible(False)
        ax[i, 0].axes.get_yaxis().set_visible(False)

        for j in range(frames_to_display):
            y_j = y[i, j, :, :] if data_format == 'channels_first' else y[i, :, :, j]
            ax[i, j + 1].imshow(y_j, cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[i, j + 1].axes.get_xaxis().set_visible(False)
            ax[i, j + 1].axes.get_yaxis().set_visible(False)
    plt.show()


def plot_error(loss_hist_file, saved_direc, plot_name):
    """Plot the training and validation error from the npz file

    Args:
        loss_hist_file: full path to .npz loss history file
        saved_direc: full path to directory where you want to save the plot
        plot_name: the name of plot
    """
    loss_history = np.load(loss_hist_file)
    loss_history = loss_history['loss_history'][()]

    err = np.subtract(1, loss_history['acc'])
    val_err = np.subtract(1, loss_history['val_acc'])

    epoch = np.arange(1, len(err) + 1, 1)
    plt.plot(epoch, err)
    plt.plot(epoch, val_err)
    plt.title('Model Error')
    plt.xlabel('Epoch')
    plt.ylabel('Model Error')
    plt.legend(['Training error', 'Validation error'], loc='upper right')

    filename = os.path.join(saved_direc, plot_name)
    plt.savefig(filename, format='pdf')
