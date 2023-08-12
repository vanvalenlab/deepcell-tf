"""
TissueNet
=========

.. image:: ../../images/multiplex_overlay.png
    :width: 200pt
    :align: center

TissueNet is a training dataset for nuclear and whole cell segmentation in tissues published in
Greenwald, Miller et al. 2022.

The TissueNet dataset is composed of a train, val, and test split.

* The train split is composed of aproximately 2600 images, each of which are 512x512
  pixels. During training, we select random crops of size 256x256 from each image as
  a form of data augmentation.
* The val split is composed of aproximately 300 images, each of which is originally
  of size 512x512. However, because we do not perform any augmentation on the
  validation dataset during training, we reshape these 512x512 images into 256x256
  images so that no cropping is needed in order to pass them through the model.
  Finally, we make two copies of the val set at different image resolutions and
  concatenate them all together, resulting in a total of aproximately 3000 images
  of size 256x256,
* The test split is composed of aproximately 300 images, each of which is originally
  of size 512x512. However, because the model was trained on images that are size
  256x256, we reshape these 512x512 images into 256x256 images, resulting in
  aproximately 1200 images.

Change Log

* TissueNet 1.0 (July 2021): The original dataset used for all experiments in
  Greenwald, Miller at al.
* TissueNet 1.1 (April 2022): This version of TissueNet has gone through an additional
  round of manual QC to ensure consistency in labeling across the entire dataset.

This dataset is licensed under a `modified Apache license
<http://www.github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE>`_ for non-commercial
academic use only

The dataset can be accessed using `deepcell.datasets` with a DeepCell API key.

For more information about using a DeepCell API key, please see :doc:`/API-key`
"""

from deepcell.datasets import TissueNet

tissuenet = TissueNet(version='1.1')
X_val, y_val, meta_val = tissuenet.load_data(split='val')

# sphinx_gallery_thumbnail_path = '../images/multiplex_overlay.png'