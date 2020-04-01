# ![DeepCell Banner](https://raw.githubusercontent.com/vanvalenlab/deepcell-tf/master/docs/images/DeepCell_Logo_Banner.png)

[![Build Status](https://travis-ci.com/vanvalenlab/deepcell-tf.svg?branch=master)](https://travis-ci.com/vanvalenlab/deepcell-tf)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/deepcell-tf/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/deepcell-tf?branch=master)
[![Documentation Status](https://img.shields.io/readthedocs/deepcell?logo=Read-the-Docs)](https://deepcell.readthedocs.io/en/master)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE)

DeepCell-tf is a neural network library for single-cell analysis, written in Python and built using [TensorFlow](https://github.com/tensorflow/tensorflow) and [Keras](https://www.tensorflow.org/guide/keras). It is the heart of the DeepCell ecosystem.

DeepCell aids in biological analysis by automatically segmenting and classifying cells in optical microscopy images.  The framework processes raw images and uniquely annotates each cell in the image.  These annotations can be used to quantify a variety of cellular properties.

Read the documentation at [deepcell.readthedocs.io](https://deepcell.readthedocs.io).

For more information on deploying DeepCell in the cloud refer to the [DeepCell Kiosk documentation](https://deepcell-kiosk.readthedocs.io).

## Examples

<table width="700" border="1" cellpadding="5">

<tr>
<td align="center" valign="center">
Raw Image
</td>

<td align="center" valign="center">
Tracked Image
</td>
</tr>

<tr>
<td align="center" valign="center">
<img src="https://raw.githubusercontent.com/vanvalenlab/deepcell-tf/master/docs/images/raw.gif" alt="Raw Image" />
</td>

<td align="center" valign="center">
<img src="https://raw.githubusercontent.com/vanvalenlab/deepcell-tf/master/docs/images/tracked.gif" alt="Tracked Image" />
</td>
</tr>

</table>

## Getting Started

The fastest way to get started with DeepCell-tf is to run one of our existing images from Docker Hub. To do so, run the following command:

```bash
# Start a GPU enabled container on two GPUs
docker run --gpus '"device=1,2"' -it --rm -p 8888:8888 vanvalenlab/deepcell-tf:0.4.0-gpu
```

This will start a Jupyter session (using the default port 8888), with several example notebooks detailing various training methods. The default port can be changed to any non-reserved port by updating `-p 8888:8888` to, e.g., `-p 81:8888`. If you run accross any errors getting this started, you may be missing some neccesary dependencies. If so, please refer to the DeepCell-tf for Developers section.

For examples of how to use the DeepCell-tf library, check out the following notebooks:

- [Training a segmentation model](https://deepcell.readthedocs.io/en/master/Training-Segmentation.html)
- [Training a tracking model](https://deepcell.readthedocs.io/en/master/Training-Tracking.html)

## DeepCell Applications and DeepCell Datasets

Together <tt><a href="https://deepcell.readthedocs.io/en/master/API/deepcell.datasets.html">deepcell.datasets</a></tt> and <tt><a href="https://deepcell.readthedocs.io/en/master/API/deepcell.applications.html">deepcell.applications</a></tt> provide an entrypoint to deep learning for biologists. The datasets module contains a variety of annotated datasets that can be used as training data. Additionally, the applications package initializes a set of models complete with the option to load pre-trained weights.

## DeepCell-tf for Developers

DeepCell-tf uses `nvidia-docker` and `tensorflow` to enable GPU processing. If using GCP, there are [pre-built images](https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/nvidia_gpu_cloud_image) which come with CUDA, docker, and nvidia-docker pre-installed. Otherwise, you will need to install [docker](https://docs.docker.com/install/linux/docker-ce/debian/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), and [CUDA](https://developer.nvidia.com/cuda-downloads) separately.

### Build a local docker container, specifying the tensorflow version with TF_VERSION

```bash
git clone https://github.com/vanvalenlab/deepcell-tf.git
cd deepcell-tf
docker build --build-arg TF_VERSION=1.15.0-gpu -t $USER/deepcell-tf .
```

### Run the new docker image

```bash
# '"device=0"' refers to the specific GPU(s) to run DeepCell-tf on, and is not required
docker run --gpus '"device=0"' -it \
-p 8888:8888 \
$USER/deepcell-tf:latest
```

It can also be helpful to mount the local copy of the repository and the scripts to speed up local development. However, if you are going to mount a local version of the repository, you must first run the docker image without the local repository mounted so that the C extensions can be compiled and then copied over to your local version.

```bash
# First run the docker image without mounting externally
docker run --gpus '"device=0"' -it \
-p 8888:8888 \
$USER/deepcell-tf:latest

# Use ctrl-p, ctrl-q (or ctrl+p+q) to exit the running docker image without shutting it down

# Then, get the container_id corresponding to the running image of DeepCell-tf
container_id=$(docker ps -q --filter ancestor="$USER/deepcell-tf")

# Copy the compiled c extensions into your local version of the codebase:
docker cp "$container_id:/usr/local/lib/python3.6/dist-packages/deepcell/utils/compute_overlap.cpython-36m-x86_64-linux-gnu.so" deepcell/utils/compute_overlap.cpython-36m-x86_64-linux-gnu.so

# close the running docker
docker kill $container_id

# you can now start the docker image with the code mounted for easy editing
docker run --gpus '"device=0"' -it \
    -p 8888:8888 \
    -v $PWD/deepcell:/usr/local/lib/python3.6/dist-packages/deepcell/ \
    -v $PWD/scripts:/notebooks \
    -v /$PWD:/data \
    $USER/deepcell-tf:latest
```

## How to Cite

- [The original DeepCell paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005177)
- [Dynamic allocation of computational resources for deep learning-enabled cellular image analysis with Kubernetes](https://www.biorxiv.org/content/10.1101/505032v3)

## Copyright

Copyright © 2016-2020 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Paul Allen Family Foundation, Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
All rights reserved.

## License

This software is licensed under a modified [APACHE2](https://github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE). See [LICENSE](https://github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE) for full details.

## Trademarks

All other trademarks referenced herein are the property of their respective owners.

## Credits

[![Van Valen Lab, Caltech](https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg)](http://www.vanvalen.caltech.edu/)
