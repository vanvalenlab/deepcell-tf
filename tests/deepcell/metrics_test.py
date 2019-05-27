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
"""Tests for metrics.py accuracy statistics"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import datetime
from random import sample

import numpy as np
import pandas as pd
from skimage.measure import label
from tensorflow.python.platform import test

from deepcell import metrics


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def _get_image_multichannel(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h, 2) * 64
    variance = np.random.rand(img_w, img_h, 2) * (255 - 64)
    img = np.random.rand(img_w, img_h, 2) * variance + bias
    return img


def _generate_test_masks():
    img_w = img_h = 30
    mask_images = []
    for _ in range(8):
        imarray = np.random.randint(2, size=(img_w, img_h, 1))
        mask_images.append(imarray)
    return mask_images


def _generate_stack_3d():
    img_w = img_h = 30
    imarray = np.random.randint(0, high=2, size=(5, img_w, img_h))
    return imarray


def _generate_stack_4d():
    img_w = img_h = 30
    imarray = np.random.randint(0, high=2, size=(5, img_w, img_h, 2))
    return imarray


def _generate_df():
    df = pd.DataFrame(np.random.rand(8, 4))
    return df


def _sample1(w, h, imw, imh, merge):
    """Basic two cell merge/split"""
    x = np.random.randint(0, imw - w * 2)
    y = np.random.randint(0, imh - h * 2)

    im = np.zeros((imw, imh))
    im[0:2, 0:2] = 1
    im[x:x + w, y:y + h] = 2
    im[x + w:x + 2 * w, y:y + h] = 3

    # Randomly rotate to pick horizontal or vertical
    if np.random.random() > 0.5:
        im = np.rot90(im)

    if merge:
        # Return merge error
        pred = im.copy()
        pred[pred == 3] = 2
        return im.astype('int'), pred.astype('int')
    else:
        # Return split error
        true = im.copy()
        true[true == 3] = 2
        return true.astype('int'), im.astype('int')


def _sample2(w, h, imw, imh):
    """Merge of three cells"""
    x = np.random.randint(0, imw - w)
    y = np.random.randint(0, imh - h)

    # Determine split points
    xs = np.random.randint(1, w * 0.9)
    ys = np.random.randint(1, h * 0.9)

    im = np.zeros((imw, imh))
    im[0:2, 0:2] = 4
    im[x:x + xs, y:y + ys] = 1
    im[x + xs:x + w, y:y + ys] = 2
    im[x:x + w, y + ys:y + h] = 3

    return im


def _sample2_2merge(w, h, imw, imh):

    im = _sample2(w, h, imw, imh)

    a, b = sample(set([1, 2, 3]), 2)
    pred = im.copy()
    pred[pred == b] = a

    return im.astype('int'), pred.astype('int')


def _sample2_3merge(w, h, imw, imh):

    im = _sample2(w, h, imw, imh)

    pred = (im != 0).copy()
    pred[0:2, 0:2] = 2

    return im.astype('int'), pred.astype('int')


def _sample3(w, h, imw, imh):
    """Wrong boundaries for 3 call clump"""

    x = np.random.randint(0, imw - w)
    y = np.random.randint(0, imh - h)

    # Determine split points
    xs = np.random.randint(1, w * 0.9)
    ys = np.random.randint(1, h * 0.9)

    im = np.zeros((imw, imh))
    im[x:x + xs, y:y + ys] = 1
    im[x + xs:x + w, y:y + ys] = 2
    im[x:x + w, y + ys:y + h] = 3

    true = im

    xs = np.random.randint(1, w * 0.9)
    ys = np.random.randint(1, h * 0.9)
    im = np.zeros((imw, imh))
    im[x:x + xs, y:y + ys] = 1
    im[x + xs:x + w, y:y + ys] = 2
    im[x:x + w, y + ys:y + h] = 3

    pred = im

    return true.astype('int'), pred.astype('int')


class MetricFunctionsTest(test.TestCase):

    def test_pixelstats_output(self):
        y_true = _get_image()
        y_pred = _get_image()

        out2 = metrics.stats_pixelbased(y_true, y_pred)
        self.assertIsInstance(out2, dict)

        # Test mistmatch size error
        self.assertRaises(ValueError, metrics.stats_pixelbased,
                          np.ones((10, 10)), np.ones((20, 20)))

    def test_split_stack(self):
        # Test batch True condition
        arr = np.ones((10, 100, 100, 1))
        out = metrics.split_stack(arr, True, 10, 1, 10, 2)
        outshape = (10 * 10 * 10, 100 / 10, 100 / 10, 1)
        self.assertEqual(outshape, out.shape)

        # Test batch False condition
        arr = np.ones((100, 100, 1))
        out = metrics.split_stack(arr, False, 10, 0, 10, 1)
        outshape = (10 * 10, 100 / 10, 100 / 10, 1)
        self.assertEqual(outshape, out.shape)

        # Test splitting in only one axis
        out = metrics.split_stack(arr, False, 10, 0, 1, 1)
        outshape = (10 * 1, 100 / 10, 100 / 1, 1)
        self.assertEqual(outshape, out.shape)

        out = metrics.split_stack(arr, False, 1, 0, 10, 1)
        outshape = (10 * 1, 100 / 1, 100 / 10, 1)
        self.assertEqual(outshape, out.shape)

        # Raise errors for uneven division
        self.assertRaises(ValueError, metrics.split_stack, arr, False, 11, 0, 10, 1)
        self.assertRaises(ValueError, metrics.split_stack, arr, False, 10, 0, 11, 1)


class TestMetricsObject(test.TestCase):

    def test_Metrics_init(self):
        m = metrics.Metrics('test')

        self.assertEqual(hasattr(m, 'output'), True)

    def test_all_pixel_stats(self):
        m = metrics.Metrics('test')

        before = len(m.output)

        y_true = _generate_stack_4d()
        y_pred = _generate_stack_4d()

        m.all_pixel_stats(y_true, y_pred)

        # Check that items were added to output
        self.assertNotEqual(before, len(m.output))

        # Check mismatch error
        self.assertRaises(ValueError, m.all_pixel_stats, np.ones(
            (10, 10, 10, 1)), np.ones((5, 5, 5, 1)))

    def test_df_to_dict(self):
        m = metrics.Metrics('test')
        df = _generate_df()

        L = m.pixel_df_to_dict(df)

        # Check output types
        self.assertNotEqual(len(L), 0)
        self.assertIsInstance(L, list)
        self.assertIsInstance(L[0], dict)

    def test_confusion_matrix(self):
        y_true = _generate_stack_4d()
        y_pred = _generate_stack_4d()

        m = metrics.Metrics('test')

        cm = m.calc_pixel_confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape[0], y_true.shape[-1])

    def test_metric_object_stats(self):
        y_true = label(_generate_stack_3d())
        y_pred = label(_generate_stack_3d())

        m = metrics.Metrics('test')
        before = len(m.output)

        m.calc_object_stats(y_true, y_pred)

        # Check data added to output
        self.assertNotEqual(before, len(m.output))

    def test_save_to_json(self):
        name = 'test'
        outdir = self.get_temp_dir()
        m = metrics.Metrics(name, outdir=outdir)

        # Create test list to save
        L = []
        for i in range(10):
            L.append(dict(
                name=i,
                value=i,
                feature='test',
                stat_type='output'
            ))

        m.save_to_json(L)
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        outfilename = os.path.join(outdir, name + '_' + todays_date + '.json')

        # Check that file exists
        self.assertEqual(os.path.isfile(outfilename), True)

        # Check that it can be opened
        with open(outfilename) as json_file:
            data = json.load(json_file)

        # Check data types from loaded data
        self.assertIsInstance(data, dict)
        self.assertItemsEqual(list(data.keys()), ['metrics', 'metadata'])
        self.assertIsInstance(data['metrics'], list)
        self.assertIsInstance(data['metadata'], dict)

    def test_run_all(self):
        y_true_lbl = label(_generate_stack_3d())
        y_pred_lbl = label(_generate_stack_3d())
        y_true_unlbl = _generate_stack_4d()
        y_pred_unlbl = _generate_stack_4d()

        name = 'test'
        outdir = self.get_temp_dir()
        m = metrics.Metrics(name, outdir=outdir)

        before = len(m.output)

        m.run_all(y_true_lbl, y_pred_lbl, y_true_unlbl, y_pred_unlbl)

        # Assert that data was added to output
        self.assertNotEqual(len(m.output), before)

        # Check output file
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        outname = os.path.join(outdir, name + '_' + todays_date + '.json')
        self.assertEqual(os.path.isfile(outname), True)


class TestObjectAccuracy(test.TestCase):

    def test_init(self):
        y_true, _ = _sample1(10, 10, 30, 30, True)

        # Test basic initialization
        o = metrics.ObjectAccuracy(y_true, y_true, test=True)

        # Check that object numbers are integers
        self.assertIsInstance(o.n_true, int)
        self.assertIsInstance(o.n_pred, int)

        self.assertEqual(o.empty_frame, False)

    def test_init_wrongsize(self):
        # Test mismatched input size
        y_true = label(_get_image())
        y_wrong = label(_get_image(img_h=200, img_w=200))
        self.assertRaises(ValueError, metrics.ObjectAccuracy, y_true, y_wrong)

    def test_init_emptyframe(self):
        y_true, y_empty = _sample1(10, 10, 30, 30, True)

        # Check handling of empty frames
        y_empty[:, :] = 0
        y_empty = y_empty.astype('int')

        oempty = metrics.ObjectAccuracy(y_true, y_empty)
        self.assertEqual(oempty.empty_frame, 'n_pred')
        oempty = metrics.ObjectAccuracy(y_empty, y_true)
        self.assertEqual(oempty.empty_frame, 'n_true')

    def test_calc_iou(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectAccuracy(y_true, y_pred, test=True)

        o._calc_iou()

        # Check that iou was created
        self.assertTrue(hasattr(o, 'iou'))

        # Check that it is not equal to initial value
        self.assertNotEqual(np.count_nonzero(o.iou), 0)

        # Test seg_thresh creation
        o = metrics.ObjectAccuracy(y_true, y_pred, test=True, seg=True)
        o._calc_iou()

        self.assertTrue(hasattr(o, 'seg_thresh'))

    def test_make_matrix(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectAccuracy(y_true, y_pred, test=True)
        o._calc_iou()

        o._make_matrix()

        self.assertTrue(hasattr(o, 'cm'))

        self.assertNotEqual(np.count_nonzero(o.cm), 0)

    def test_linear_assignment(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectAccuracy(y_true, y_pred, test=True)
        o._calc_iou()
        o._make_matrix()

        o._linear_assignment()

        cols = ['n_pred', 'n_true', 'correct_detections', 'missed_detections', 'gained_detections',
                'missed_det_from_merge', 'gained_det_from_split', 'true_det_in_catastrophe',
                'pred_det_in_catastrophe', 'merge', 'split', 'catastrophe']

        for obj in cols:
            self.assertTrue(hasattr(o, obj))

        # Test condition where seg = True
        o = metrics.ObjectAccuracy(y_true, y_pred, test=True, seg=True)
        o._calc_iou()
        o._make_matrix()
        o._linear_assignment()

        for obj in ['results', 'cm_res', 'seg_score']:
            self.assertTrue(hasattr(o, obj))

    def test_assign_loners(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectAccuracy(y_true, y_pred, test=True)
        o._calc_iou()
        o._make_matrix()
        o._linear_assignment()

        o._assign_loners()
        self.assertTrue(hasattr(o, 'cost_l_bin'))

    def test_array_to_graph(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectAccuracy(y_true, y_pred, test=True)
        o._calc_iou()
        o._make_matrix()
        o._linear_assignment()
        o._assign_loners()

        o._array_to_graph()
        self.assertTrue(hasattr(o, 'G'))

    def test_classify_graph(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        # Test that complete run through is succesful
        _ = metrics.ObjectAccuracy(y_true, y_pred)

    def test_optional_outputs(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectAccuracy(y_true, y_pred)

        o.print_report()

        df = o.save_to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)

        columns = ['n_pred', 'n_true', 'correct_detections', 'missed_detections', 'jaccard',
                   'missed_det_from_merge', 'gained_det_from_split', 'true_det_in_catastrophe',
                   'pred_det_in_catastrophe', 'merge', 'split', 'catastrophe', 'gained_detections']
        self.assertItemsEqual(columns, list(df.columns))

        # Check seg True case
        o = metrics.ObjectAccuracy(y_true, y_pred, seg=True)
        o.print_report()
        df = o.save_to_dataframe()
        columns = ['n_pred', 'n_true', 'correct_detections', 'missed_detections', 'seg', 'jaccard',
                   'missed_det_from_merge', 'gained_det_from_split', 'true_det_in_catastrophe',
                   'pred_det_in_catastrophe', 'merge', 'split', 'catastrophe', 'gained_detections']
        self.assertItemsEqual(columns, list(df.columns))
