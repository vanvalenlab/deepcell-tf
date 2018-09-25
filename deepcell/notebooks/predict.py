# Copyright 2016-2018 David Van Valen at California Institute of Technology
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
"""Automatically generate a notebook for analyzing predicted images
@author: andrewqho, willgraf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import time
import datetime

import nbformat as nbf


# Create Visual Notebook
def make_notebook(model_output,
                  output_dir=os.path.join('scripts', 'generated_notebooks')):
    """Create a visualization notebook that will help visualize
    the output of a deep learning model
    # Arguments:
        model_output: output of a deep learning model to visualize
        output_dir: directory to save the notebook
    """
    # validate inputs
    if not os.path.isfile(model_output):
        raise FileNotFoundError('{} does not exist.  '
                                '`model_output` must be a file.'.format(
                                    model_output))
    # create output_dir if it does not already exist
    try:
        os.makedirs(output_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    # list of cells that will be in the notebook
    cells = []

    # Markdown Header
    text = [
        '## Visualize Predictions'
        'A template Jupyter notebook to explore returned predictions'
    ]
    cells.append(nbf.v4.new_markdown_cell('\n'.join(text)))

    # Python imports
    imports = [
        'from matplotlib import pyplot as plt',
        'import numpy as np',
        'from PIL import Image'
    ]
    cells.append(nbf.v4.new_code_cell('\n'.join(imports)))

    # Code for Data Visualization
    visual = [
        'pil_im = Image.open("{}", "r")'.format(model_output),
        'plt.imshow(np.asarray(pil_im))'
    ]
    cells.append((nbf.v4.new_code_cell('\n'.join(visual))))

    # Create and write to new ipynb
    nb = nbf.v4.new_notebook(cells=cells)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    nbf.write(nb, os.path.join(output_dir, 'visualize_{}.ipynb'.format(ts)))
