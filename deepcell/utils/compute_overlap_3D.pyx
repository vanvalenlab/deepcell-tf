# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by David Van Valen - adapted from code by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np


def compute_overlap(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 6) ndarray of float
        b: (K, 6) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, id, box_volume
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        box_volume = (
            (query_boxes[k, 3] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 4] - query_boxes[k, 1] + 1) *
            (query_boxes[k, 5] - query_boxes[k, 2] + 1)
        )
        for n in range(N):
            id = (
                min(boxes[n, 3], query_boxes[k, 3]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if id > 0:
                iw = (
                    min(boxes[n, 4], query_boxes[k, 4]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if iw > 0:
                    ih = (
                        min(boxes[n, 5], query_boxes[k, 5]) - 
                        max(boxes[n, 2], query_boxes[k, 2]) + 1
                    )
                    if ih > 0:
                        ua = np.float64(
                            (boxes[n, 3] - boxes[n, 0] + 1) *
                            (boxes[n, 4] - boxes[n, 1] + 1) *
                            (boxes[n, 5] - boxes[n, 2] + 1) +
                            box_volume - iw * ih * id
                        )
                        overlaps[n, k] = iw * ih * id / ua
    return overlaps
