# DeepCell: Deep Learning for Single Cell Analysis

[![Build Status](https://travis-ci.org/vanvalenlab/deepcell-tf.svg?branch=master)](https://travis-ci.org/vanvalenlab/deepcell-tf)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/deepcell-tf/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/deepcell-tf?branch=master)

DeepCell is neural network library for single cell analysis, written in Python and built using [TensorFlow](https://github.com/tensorflow/tensorflow) and [Keras](https://github.com/keras-team/keras).  

DeepCell aids in biological analysis by automatically segmenting and classifying cells in optical microscopy images.  The framework processes raw images and uniquely annotates each cell in the image.  These annotations can be used to quantify a variety of cellular properties.

Read the documentaiton at [deepcell.readthedocs.io](https://deepcell.readthedocs.io)

For more information on deploying DeepCell in the cloud [refer to the DeepCell Kiosk documentation](https://deepcell-kiosk.readthedocs.io)  


## Getting Started

The fastest way to get started with DeepCell is to run the latest docker image:

```bash
nvidia-docker run -it --rm -p 8888:8888 vanvalenlab/deepcell-tf:latest
```

This will start a jupyter session, with several example notebooks detailing various training methods:

### Cell Edge and Cell Interior Segmentation

* [2D DeepCell Transform - Fully Convolutional.ipynb](scripts/deepcell/Interior-Edge%20Segmentation%202D%20Fully%20Convolutional.ipynb)

* [2D DeepCell Transform - Sample Based.ipynb](scripts/deepcell/Interior-Edge%20Segmentation%202D%20Sample%20Based.ipynb)

* [3D DeepCell Transform - Fully Convolutional.ipynb](scripts/deepcell/Interior-Edge%20Segmentation%203D%20Fully%20Convolutional.ipynb)

* [3D DeepCell Transform - Sample Based.ipynb](scripts/deepcell/Interior-Edge%20Segmentation%203D%20Sample%20Based.ipynb)

### Deep Watershed Instance Segmentation

* [2D Watershed - Fully Convolutional.ipynb](scripts/watershed/Watershed%20Transform%202D%20Fully%20Convolutional.ipynb)

* [2D Watershed - Sample Based.ipynb](scripts/watershed/Watershed%20Transform%202D%20Sample%20Based.ipynb)

* [3D Watershed - Fully Convolutional.ipynb](scripts/watershed/Watershed%20Transform%203D%20Fully%20Convolutional.ipynb)

* [3D Watershed - Sample Based.ipynb](scripts/watershed/Watershed%20Transform%203D%20Sample%20Based.ipynb)

## DeepCell for Developers

DeepCell uses `nvidia-docker` and `tensorflow` to enable GPU processing.  

### Build a local docker container

```bash
git clone https://github.com/vanvalenlab/deepcell-tf.git
cd deepcell-tf
docker build -t $USER/deepcell-tf .

```

The tensorflow version can be overridden with the build-arg `TF_VERSION`.

```bash
docker build --build-arg TF_VERSION=1.9.0-gpu -t $USER/deepcell-tf .
```

### Run the new docker image

```bash
# NV_GPU refers to the specific GPU to run DeepCell on, and is not required

# Mounting the codebase, scripts and data to the container is also optional
# but can be handy for local development

NV_GPU='0' nvidia-docker run -it \
  -p 8888:8888 \
  $USER/deepcell-tf:latest
```

It can also be helpful to mount the local copy of the repository and the scripts to speed up local development.

```bash
NV_GPU='0' nvidia-docker run -it \
  -p 8888:8888 \
  -v $PWD/deepcell:/usr/local/lib/python3.5/dist-packages/deepcell/ \
  -v $PWD/scripts:/notebooks \
  -v /data:/data \
  $USER/deepcell-tf:latest
```

## Copyright

Copyright © 2016-2019 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Paul Allen Family Foundation, Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.  
All rights reserved.

## License 

This software is licensed under a modified [APACHE2](LICENSE).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

See [LICENSE](LICENSE) for full details.

## Trademarks

All other trademarks referenced herein are the property of their respective owners.

## Credits

[![Van Valen Lab, Caltech](https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg)](http://www.vanvalen.caltech.edu/)
