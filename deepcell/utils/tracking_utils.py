# Copyright 2016-2019 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Utilities for tracking cells"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tarfile
import pathlib
import tempfile
from io import BytesIO

import numpy as np
from tensorflow.python.keras import backend as K

from deepcell.utils.misc_utils import sorted_nicely


def count_pairs(y, same_probability=0.5, data_format=None):
    """Compute number of training samples needed to observe all cell pairs.

    Args:
        y: 5D tensor of cell labels
        same_probability: liklihood that 2 cells are the same

    Returns:
        the total pairs needed to sample to see all possible pairings
    """
    if data_format is None:
        data_format = K.image_data_format()

    total_pairs = 0
    zaxis = 2 if data_format == 'channels_first' else 1
    for b in range(y.shape[0]):
        # count the number of cells in each image of the batch
        cells_per_image = []
        for f in range(y.shape[zaxis]):
            if data_format == 'channels_first':
                num_cells = len(np.unique(y[b, :, f, :, :]))
            else:
                num_cells = len(np.unique(y[b, f, :, :, :]))
            cells_per_image.append(num_cells)

        # Since there are many more possible non-self pairings than there
        # are self pairings, we want to estimate the number of possible
        # non-self pairings and then multiply that number by two, since the
        # odds of getting a non-self pairing are 50%, to find out how many
        # pairs we would need to sample to (statistically speaking) observe
        # all possible cell-frame pairs. We're going to assume that the
        # average cell is present in every frame. This will lead to an
        # underestimate of the number of possible non-self pairings, but it
        # is unclear how significant the underestimate is.
        average_cells_per_frame = sum(cells_per_image) // y.shape[zaxis]
        non_self_cellframes = (average_cells_per_frame - 1) * y.shape[zaxis]
        non_self_pairings = non_self_cellframes * max(cells_per_image)

        # Multiply cell pairings by 2 since the
        # odds of getting a non-self pairing are 50%
        cell_pairings = non_self_pairings / same_probability
        # Add this batch cell-pairings to the total count
        total_pairs += cell_pairings
    return total_pairs


def load_trks(trks_file):
    """Load a trks_file.

        Args:
            trks_file: full path to the file

        Returns:
            A dictionary with raw, tracked, and lineage data
    """
    with tarfile.open(trks_file, 'r') as trks:
        # trks.extractfile opens a file in bytes mode, json can't use bytes.
        trk_data = trks.getmember('lineages.json')
        lineages = json.loads(trks.extractfile(trk_data).read().decode())

        # numpy can't read these from disk...
        array_file = BytesIO()
        array_file.write(trks.extractfile('raw.npy').read())
        array_file.seek(0)
        raw = np.load(array_file)
        array_file.close()

        array_file = BytesIO()
        array_file.write(trks.extractfile('tracked.npy').read())
        array_file.seek(0)
        tracked = np.load(array_file)
        array_file.close()

    # JSON only allows strings as keys, so we convert them back to ints here
    for i, tracks in enumerate(lineages):
        lineages[i] = {int(k): v for k, v in tracks.items()}

    return {'lineages': lineages, 'X': raw, 'y': tracked}


def load_trk(filename):
    """Load a trk_file.

        Args:
            filename: full path to the file

        Returns:
            A dictionary with raw, tracked, and lineage data
    """
    with tarfile.open(filename, "r") as trks:
        # trks.extractfile opens a file in bytes mode, json can't use bytes.
        lineage = json.loads(
                trks.extractfile(
                    trks.getmember("lineage.json")).read().decode())

        # numpy can't read these from disk...
        array_file = BytesIO()
        array_file.write(trks.extractfile("raw.npy").read())
        array_file.seek(0)
        raw = np.load(array_file)
        array_file.close()

        array_file = BytesIO()
        array_file.write(trks.extractfile("tracked.npy").read())
        array_file.seek(0)
        tracked = np.load(array_file)
        array_file.close()

    # JSON only allows strings as keys, so we convert them back to ints here
    lineage = {int(k): v for k, v in lineage.items()}

    return {"lineage": lineage, "raw": raw, "tracked": tracked}


def trk_folder_to_trks(dirname, trks_filename):
    """Compiles a directory of trk files into one trks_file.

        Args:
            dirname: full path to the directory containing multiple trk files
            trks_filename: desired filename (the name should end in .trks)

        Returns:
            Nothing
    """
    lineages = []
    raw = []
    tracked = []

    file_list = os.listdir(dirname)
    file_list_sorted = sorted_nicely(file_list)

    for filename in file_list_sorted:
        trk = load_trk(os.path.join(dirname, filename))
        lineages.append(trk["lineage"])
        raw.append(trk["raw"])
        tracked.append(trk["tracked"])

    file_path = os.path.join(os.path.dirname(dirname), trks_filename)

    save_trks(file_path, lineages, raw, tracked)


def save_trks(filename, lineages, raw, tracked):
    """Saves raw, tracked, and lineage data into one trks_file.

        Args:
            filename: full path to the the final trk files
            lineages: a list of dictionaries saved as a json
            raw: raw images data
            tracked: annotated image data

        Returns:
            Nothing
    """
    if not filename.endswith(".trks"):
        raise ValueError("filename must end with '.trks'")

    with tarfile.open(filename, "w") as trks:
        with tempfile.NamedTemporaryFile("w") as lineages_file:
            json.dump(lineages, lineages_file, indent=1)
            lineages_file.flush()
            trks.add(lineages_file.name, "lineages.json")

        with tempfile.NamedTemporaryFile() as raw_file:
            np.save(raw_file, raw)
            raw_file.flush()
            trks.add(raw_file.name, "raw.npy")

        with tempfile.NamedTemporaryFile() as tracked_file:
            np.save(tracked_file, tracked)
            tracked_file.flush()
            trks.add(tracked_file.name, "tracked.npy")