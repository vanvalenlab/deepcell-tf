# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
"""Tests for Mesmer Application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np

from unittest.mock import Mock

from tensorflow.python.platform import test

from deepcell.model_zoo import PanopticNet
from deepcell.applications import Mesmer
from deepcell.applications import MultiplexSegmentation
from deepcell.applications.mesmer import format_output_mesmer
from deepcell.applications.mesmer import mesmer_postprocess
from deepcell.applications.mesmer import mesmer_preprocess


# test pre- and post-processing functions
def test_mesmer_preprocess():
    height, width = 300, 300
    img = np.random.randint(0, 100, (height, width))

    # make rank 4 (batch, X, y, channel)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # single bright spot
    img[0, 200, 200, 0] = 5000

    # histogram normalized
    processed = mesmer_preprocess(img)
    assert (processed <= 1).all() and (processed >= -1).all()

    # maxima is no longer significantly greater than rest of image
    new_spot_val = processed[0, 200, 200, 0]
    processed[0, 200, 200, 0] = 0.5
    next_max_val = np.max(processed)

    # difference between bright spot and next greatest value is essentially nothing
    assert np.round(new_spot_val / next_max_val, 1) == 1

    # histogram normalization without thresholding
    processed_hist = mesmer_preprocess(img, threshold=False)
    assert (processed_hist <= 1).all() and (processed_hist >= -1).all()

    new_spot_val = processed_hist[0, 200, 200, 0]
    processed_hist[0, 200, 200, 0] = 0.5
    next_max_val = np.max(processed_hist)
    assert np.round(new_spot_val / next_max_val, 1) > 1

    # thresholding without histogram normalization
    processed_thresh = mesmer_preprocess(img, normalize=False)
    assert not (processed_thresh <= 1).all()

    new_spot_val = processed_thresh[0, 200, 200, 0]
    processed_thresh[0, 200, 200, 0] = 0.5
    next_max_val = np.max(processed_thresh)
    assert np.round(new_spot_val / next_max_val, 1) == 1

    # no change to image
    not_processed = mesmer_preprocess(img, normalize=False, threshold=False)
    assert np.all(not_processed == img)

    # bad input
    with pytest.raises(ValueError):
        _ = mesmer_preprocess(np.zeros((3, 50, 50)))


def test_mesmer_postprocess(mocker):
    # create dict, with each image having a different constant value
    base_array = np.ones((1, 20, 20, 1))

    whole_cell_list = [base_array * mult for mult in range(1, 3)]

    nuclear_list = [base_array * mult for mult in range(3, 5)]

    model_output = {'whole-cell': whole_cell_list, 'nuclear': nuclear_list}

    # whole cell predictions only
    whole_cell = mesmer_postprocess(model_output=model_output,
                                    compartment='whole-cell')
    assert whole_cell.shape == (1, 20, 20, 1)

    # nuclear predictions only
    nuclear = mesmer_postprocess(model_output=model_output,
                                 compartment='nuclear')
    assert nuclear.shape == (1, 20, 20, 1)

    # both whole-cell and nuclear predictions
    both = mesmer_postprocess(model_output=model_output,
                              compartment='both')
    assert both.shape == (1, 20, 20, 2)

    # make sure correct arrays are being passed to helper function
    def mock_deep_watershed(model_output):
        pixelwise_interior_vals = model_output[-1]
        return pixelwise_interior_vals

    mocker.patch('deepcell.applications.mesmer.deep_watershed',
                 mock_deep_watershed)

    # whole cell predictions only
    whole_cell_mocked = mesmer_postprocess(model_output=model_output,
                                           compartment='whole-cell')

    assert np.array_equal(whole_cell_mocked, whole_cell_list[1])

    # nuclear predictions only
    whole_cell_mocked = mesmer_postprocess(model_output=model_output,
                                           compartment='nuclear')

    assert np.array_equal(whole_cell_mocked, nuclear_list[1])

    with pytest.raises(ValueError):
        whole_cell = mesmer_postprocess(model_output=model_output,
                                        compartment='invalid')


def test_format_output_mesmer():

    # create output list, each with a different constant value across image
    base_array = np.ones((1, 20, 20, 1))

    whole_cell_list = [base_array * mult for mult in range(1, 5)]
    whole_cell_list = [whole_cell_list[0],
                       np.concatenate(whole_cell_list[1:4], axis=-1)]

    # create output list for nuclear predictions
    nuclear_list = [img * 2 for img in whole_cell_list]

    combined_list = whole_cell_list + nuclear_list

    output = format_output_mesmer(combined_list)

    assert set(output.keys()) == {'whole-cell', 'nuclear'}

    assert np.array_equal(output['whole-cell'][0], base_array)
    assert np.array_equal(output['nuclear'][0], base_array * 2)

    assert np.array_equal(output['whole-cell'][1], base_array * 3)
    assert np.array_equal(output['nuclear'][1], base_array * 6)

    with pytest.raises(ValueError):
        output = format_output_mesmer(combined_list[:3])


# test application
class TestMesmer(test.TestCase):

    def test_mesmer_app(self):
        with self.cached_session():
            whole_cell_classes = [1, 3]
            nuclear_classes = [1, 3]
            num_semantic_classes = whole_cell_classes + nuclear_classes
            num_semantic_heads = len(num_semantic_classes)

            model = PanopticNet(
                'resnet50',
                input_shape=(256, 256, 2),
                norm_method=None,
                num_semantic_heads=num_semantic_heads,
                num_semantic_classes=num_semantic_classes,
                location=True,
                include_top=True,
                use_imagenet=False)

            app = Mesmer(model)

            # test output shape
            shape = app.model.output_shape
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 4)

            # test predict with default
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x)
            self.assertEqual(x.shape[:-1], y.shape[:-1])

            # test predict with nuclear compartment only
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x, compartment='nuclear')
            self.assertEqual(x.shape[:-1], y.shape[:-1])
            self.assertEqual(y.shape[-1], 1)

            # test predict with cell compartment only
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x, compartment='whole-cell')
            self.assertEqual(x.shape[:-1], y.shape[:-1])
            self.assertEqual(y.shape[-1], 1)

            # test predict with both cell and nuclear compartments
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x, compartment='both')
            self.assertEqual(x.shape[:-1], y.shape[:-1])
            self.assertEqual(y.shape[-1], 2)

            # test that kwargs are passed through successfully
            app._predict_segmentation = Mock()

            # get defaults
            _ = app.predict(x, compartment='whole-cell')
            args = app._predict_segmentation.call_args[1]
            default_cell_kwargs = args['postprocess_kwargs']['whole_cell_kwargs']
            default_nuc_kwargs = args['postprocess_kwargs']['nuclear_kwargs']

            # change one of the args for each compartment
            maxima_threshold_cell = default_cell_kwargs['maxima_threshold'] + 0.1
            radius_nuc = default_nuc_kwargs['radius'] + 2

            _ = app.predict(x, compartment='whole-cell',
                            postprocess_kwargs_whole_cell={'maxima_threshold':
                                                           maxima_threshold_cell},
                            postprocess_kwargs_nuclear={'radius': radius_nuc})

            args = app._predict_segmentation.call_args[1]
            cell_kwargs = args['postprocess_kwargs']['whole_cell_kwargs']
            assert cell_kwargs['maxima_threshold'] == maxima_threshold_cell

            nuc_kwargs = args['postprocess_kwargs']['nuclear_kwargs']
            assert nuc_kwargs['radius'] == radius_nuc

            # check that rest are unchanged
            cell_kwargs['maxima_threshold'] = default_cell_kwargs['maxima_threshold']
            assert cell_kwargs == default_cell_kwargs

            nuc_kwargs['radius'] = default_nuc_kwargs['radius']
            assert nuc_kwargs == default_nuc_kwargs

            # test legacy version
            old_app = MultiplexSegmentation(model)

            # test predict with default
            x = np.random.rand(1, 500, 500, 2)
            y = old_app.predict(x)
            self.assertEqual(x.shape[:-1], y.shape[:-1])
