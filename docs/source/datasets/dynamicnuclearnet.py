"""
DynamicNuclearNet
=================

.. image:: ../../images/tracked.gif
    :width: 200pt
    :align: center

DynamicNuclearNet is a training dataset for nuclear segmentation and tracking published in
Schwartz et al. 2023. The dataset is made up of two subsets, one for tracking and one for
segmentation.

This dataset is licensed under a `modified Apache license
<http://www.github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE>`_ for non-commercial
academic use only

The dataset can be accessed using `deepcell.datasets` with a DeepCell API key.

For more information about using a DeepCell API key, please see :doc:`/API-key`

Tracking
^^^^^^^^

Each batch of the dataset contains three components

* X: raw fluorescent nuclear data
* y: nuclear segmentation masks
* lineages: lineage records including the cell id, frames present and division links from
  parent to daughter cells

"""

from deepcell.datasets import DynamicNuclearNetTracking
# sphinx_gallery_thumbnail_path = '../images/tracked.gif'

dnn_trk = DynamicNuclearNetTracking(version='1.0')
X_val, y_val, lineage_val = dnn_trk.load_data(split='val')
data_source = dnn_trk.load_source_metadata()

#%%
# Segmentation
# ^^^^^^^^^^^^
# Each batch of the dataset includes three components
#
# * X: raw fluorescent nuclear data
# * y: nuclear segmentation masks
# * metadata: description of the source of each batch

from deepcell.datasets import DynamicNuclearNetSegmentation

dnn_seg = DynamicNuclearNetSegmentation(version='1.0')
X_val, y_val, meta_val = dnn_seg.load_data(split='val')
