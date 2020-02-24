.. _README:

DeepCell: Deep Learning for Single Cell Analysis
================================================

.. image:: https://travis-ci.com/vanvalenlab/deepcell-tf.svg?branch=master
    :target: https://travis-ci.com/vanvalenlab/deepcell-tf
    :alt: Build Status

.. image:: https://coveralls.io/repos/github/vanvalenlab/deepcell-tf/badge.svg?branch=master
    :target: https://coveralls.io/github/vanvalenlab/deepcell-tf?branch=master
    :alt: Coverage Status

.. image:: https://img.shields.io/readthedocs/deepcell?logo=Read-the-Docs
    :target: https://deepcell.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status


DeepCell is neural network library for single cell analysis, written in Python and built using `TensorFlow <https://github.com/tensorflow/tensorflow>`_ and `Keras <https://www.tensorflow.org/guide/keras>`_.

DeepCell aids in biological analysis by automatically segmenting and classifying cells in optical microscopy images.  The framework processes raw images and uniquely annotates each cell in the image.  These annotations can be used to quantify a variety of cellular properties.

Read the documentation at `deepcell.readthedocs.io <https://deepcell.readthedocs.io>`_

For more information on deploying DeepCell in the cloud refer to the `DeepCell Kiosk documentation <https://deepcell-kiosk.readthedocs.io>`_.

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

For examples of how to use the deepcell library, check out the following notebooks:
* `Training a segmentation model <https://deepcell.readthedocs.io/en/master/Training-Segmentation.html>`_
* `Training a tracking model <https://deepcell.readthedocs.io/en/master/Training-Tracking.html>`_

``deepcell.applications`` and ``deepcell.datasets``
---------------------------------------------------------
Together |deepcell.datasets| and |deepcell.applications|  work together to provide an accessible entrypoint to deep learning for biologists. The datasets module contains a variety of annotated datasets that can be used as training data. Additionally, the applications package initializes a set of models complete with the option to initialize with pre-trained weights.

.. |deepcell.datasets| raw:: html

    <tt><a href="https://deepcell.readthedocs.io/en/master/API/deepcell.datasets.html">deepcell.datasets</a></tt>

.. |deepcell.applications| raw:: html

    <tt><a href="https://deepcell.readthedocs.io/en/mrgn-docs/API/deepcell.applications.html">deepcell.applications</a></tt>

DeepCell for Developers
-----------------------

DeepCell uses ``nvidia-docker`` and ``tensorflow`` to enable GPU processing. If using GCP, there are `pre-built images <https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/nvidia_gpu_cloud_image>`_ which come with CUDA, docker, and nvidia-docker pre-installed. Otherwise, you will need to install `docker <
https://docs.docker.com/install/linux/docker-ce/debian/>`_, `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_, and `CUDA <https://developer.nvidia.com/cuda-downloads>`_ separately.

Build a local docker container, specifying the tensorflow version with TF_VERSION
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/vanvalenlab/deepcell-tf.git
    cd deepcell-tf
    docker build --build-arg TF_VERSION=1.15.0-gpu -t $USER/deepcell-tf .


Run the new docker image
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # NV_GPU refers to the specific GPU to run DeepCell on, and is not required

    NV_GPU='0' nvidia-docker run -it \
    -p 8888:8888 \
    $USER/deepcell-tf:latest

It can also be helpful to mount the local copy of the repository and the scripts to speed up local development. However, if you are going to mount a local version of the repository, you must first run the docker image without the local repository mounted so that the c extensions can be compiled and then copied over to your local version.

.. code-block:: bash

    # First run the docker image without mounting externally
    NV_GPU='0' nvidia-docker run -it \
    -p 8888:8888 \
    $USER/deepcell-tf:latest

    # Use ctrl-p, ctrl-q to exit the running docker image without shutting it down

    # Then, get the container_id corresponding to the running image of deepcell
    container_id=$(docker ps -q --filter ancestor="$USER/deepcell-tf")

    # Copy the compiled c extensions into your local version of the codebase:
    docker cp "$container_id:/usr/local/lib/python3.6/dist-packages/deepcell/utils/compute_overlap.cpython-36m-x86_64-linux-gnu.so" deepcell/utils/compute_overlap.cpython-36m-x86_64-linux-gnu.so

    # close the running docker
    docker kill $container_id

    # you can now start the docker image with the code mounted for easy editing
    NV_GPU='0' nvidia-docker run -it \
    -p 8888:8888 \
    -v $PWD/deepcell:/usr/local/lib/python3.6/dist-packages/deepcell/ \
    -v $PWD/scripts:/notebooks \
    -v /$PWD:/data \
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
    :alt: Van Valen Lab. Caltech
