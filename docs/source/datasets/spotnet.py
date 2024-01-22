"""
SpotNet
=======

.. image:: ../../images/spots.png
    :width: 200pt
    :align: center

SpotNet is a training dataset for a deep learning model for spot detection published in
Laubscher et al. 2023. 

This dataset is licensed under a `modified Apache license
<http://www.github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE>`_ for non-commercial
academic use only.

The dataset can be accessed using `deepcell.datasets` with a DeepCell API key.

For more information about using a DeepCell API key, please see :doc:`/API-key`.

Each batch of the dataset contains two components:

* X: raw images of fluorescent spots
* y: coordinate annotations for spot locations

"""

from deepcell.datasets import SpotNet

spotnet = SpotNet(version='1.1')
X_val, y_val = spotnet.load_data(split='val')
# sphinx_gallery_thumbnail_path = '../images/spots.png'
