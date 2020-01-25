.. _README:

DeepCell: Deep Learning for Single Cell Analysis
================================================

.. image:: https://travis-ci.com/vanvalenlab/deepcell-tf.svg?branch=master
    :target: https://travis-ci.com/vanvalenlab/deepcell-tf

.. image:: https://coveralls.io/repos/github/vanvalenlab/deepcell-tf/badge.svg?branch=master
    :target: https://coveralls.io/github/vanvalenlab/deepcell-tf?branch=master


DeepCell is neural network library for single cell analysis, written in Python and built using `TensorFlow <https://github.com/tensorflow/tensorflow>`_ and `Keras <https://www.tensorflow.org/guide/keras>`_.

DeepCell aids in biological analysis by automatically segmenting and classifying cells in optical microscopy images.  The framework processes raw images and uniquely annotates each cell in the image.  These annotations can be used to quantify a variety of cellular properties.

Read the documentation at `deepcell.readthedocs.io <https://deepcell.readthedocs.io>`_

For more information on deploying DeepCell in the cloud `refer to the DeepCell Kiosk documentation <https://deepcell-kiosk.readthedocs.io>`_.

Examples
--------

.. list-table::

    * - Raw Image
      - Segmented and Tracked
    * - .. image:: docs/images/raw.gif
      - .. image:: docs/images/tracked.gif

Getting Started
---------------

The fastest way to get started with DeepCell is to run the latest docker image:

.. code-block:: bash

    nvidia-docker run -it --rm -p 8888:8888 vanvalenlab/deepcell-tf:latest

This will start a jupyter session, with several example notebooks detailing various training methods:

PanOptic Segmentation using RetinaMask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `Retinanet Object Detection.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/feature_pyramids/RetinaNet.ipynb>`_
* `RetinaMask Instance Segmentation.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/feature_pyramids/RetinaMask.ipynb>`_
* `PanOptic Segmentation.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/feature_pyramids/PanOpticFPN.ipynb>`_

Pixel-Wise Segmentation
^^^^^^^^^^^^^^^^^^^^^^^

* `2D Pixel-Wise Transform - Fully Convolutional.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/pixelwise/Interior-Edge%20Segmentation%202D%20Fully%20Convolutional.ipynb>`_

* `2D Pixel-Wise Transform - Sample Based.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/pixelwise/Interior-Edge%20Segmentation%202D%20Sample%20Based.ipynb>`_

* `3D Pixel-Wise Transform - Fully Convolutional.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/pixelwise/Interior-Edge%20Segmentation%203D%20Fully%20Convolutional.ipynb>`_

* `3D Pixel-Wise Transform - Sample Based.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/pixelwise/Interior-Edge%20Segmentation%203D%20Sample%20Based.ipynb>`_

Deep Watershed Instance Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `2D Watershed - Fully Convolutional.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/watershed/Watershed%20Transform%202D%20Fully%20Convolutional.ipynb>`_

* `2D Watershed - Sample Based.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/watershed/Watershed%20Transform%202D%20Sample%20Based.ipynb>`_

* `3D Watershed - Fully Convolutional.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/watershed/Watershed%20Transform%203D%20Fully%20Convolutional.ipynb>`_

* `3D Watershed - Sample Based.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/watershed/Watershed%20Transform%203D%20Sample%20Based.ipynb>`_

Cell Tracking in Live Cell Imaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Training a Tracking Models.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/tracking/Training%20a%20Tracking%20Model.ipynb>`_
* `Tracking Example with Benchmarking.ipynb <https://github.com/vanvalenlab/deepcell-tf/blob/master/scripts/tracking/Tracking%20Example%20with%20Benchmarking.ipynb>`_


DeepCell for Developers
-----------------------

DeepCell uses ``nvidia-docker`` and ``tensorflow`` to enable GPU processing.

Build a local docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/vanvalenlab/deepcell-tf.git
    cd deepcell-tf
    docker build -t $USER/deepcell-tf .

The tensorflow version can be overridden with the build-arg ``TF_VERSION``.

.. code-block:: bash

    docker build --build-arg TF_VERSION=1.15.0-gpu -t $USER/deepcell-tf .


Run the new docker image
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # NV_GPU refers to the specific GPU to run DeepCell on, and is not required

    # Mounting the codebase, scripts and data to the container is also optional
    # but can be handy for local development

    NV_GPU='0' nvidia-docker run -it \
    -p 8888:8888 \
    $USER/deepcell-tf:latest

It can also be helpful to mount the local copy of the repository and the scripts to speed up local development.

.. code-block:: bash

    NV_GPU='0' nvidia-docker run -it \
    -p 8888:8888 \
    -v $PWD/deepcell:/usr/local/lib/python3.6/dist-packages/deepcell/ \
    -v $PWD/scripts:/notebooks \
    -v /data:/data \
    $USER/deepcell-tf:latest

How to Cite
-----------
* `The original DeepCell paper <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005177>`_
* `DeepCell 2.0: Automated cloud deployment of deep learning models for large-scale cellular image analysis <https://www.biorxiv.org/content/early/2018/12/22/505032.article-metrics>`_

Copyright
---------

Copyright © 2018-2020 `The Van Valen Lab <http://www.vanvalen.caltech.edu/>`_ at the California Institute of Technology (Caltech), with support from the Paul Allen Family Foundation, Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
All rights reserved.


License
-------

This software is licensed under a modified `APACHE2`_.

.. _APACHE2: https://github.com/vanvalenlab/kiosk/blob/master/LICENSE

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0

See `LICENSE`_ for full details.

.. _LICENSE: https://github.com/vanvalenlab/kiosk/blob/master/LICENSE



Trademarks
----------

All other trademarks referenced herein are the property of their respective owners.


Credits
----------

.. image:: https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg
    :target: http://www.vanvalen.caltech.edu/
