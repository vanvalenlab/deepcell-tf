# ![DeepCell Banner](https://raw.githubusercontent.com/vanvalenlab/deepcell-tf/master/docs/images/DeepCell_tf_Banner.png)

[![Build Status](https://github.com/vanvalenlab/deepcell-tf/workflows/build/badge.svg)](https://github.com/vanvalenlab/deepcell-tf/actions)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/deepcell-tf/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/deepcell-tf?branch=master)
[![Documentation Status](https://readthedocs.org/projects/deepcell/badge/?version=master)](https://deepcell.readthedocs.io/en/master/?badge=master)
[![Modified Apache 2.0](https://img.shields.io/badge/license-Modified%20Apache%202-blue)](https://github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/DeepCell.svg)](https://badge.fury.io/py/deepcell)
[![PyPi Monthly Downloads](https://img.shields.io/pypi/dm/deepcell)](https://pypistats.org/packages/deepcell)
[![Python Versions](https://img.shields.io/pypi/pyversions/deepcell.svg)](https://pypi.org/project/deepcell/)

`deepcell-tf` is a deep learning library for single-cell analysis of biological images. It is written in Python and built using [TensorFlow 2](https://github.com/tensorflow/tensorflow).

This library allows users to apply pre-existing models to imaging data as well as to develop new deep learning models for single-cell analysis. This library specializes in models for cell segmentation (whole-cell and nuclear) in 2D and 3D images as well as cell tracking in 2D time-lapse datasets. These models are applicable to data ranging from multiplexed images of tissues to dynamic live-cell imaging movies.

`deepcell-tf` is one of several resources created by the [Van Valen lab](http://vanvalen.caltech.edu/) to facilitate the development and application of new deep learning methods to biology. Other projects within our DeepCell ecosystem include the [DeepCell Toolbox](https://github.com/vanvalenlab/deepcell-toolbox) for pre and post-processing the outputs of deep learning models, [DeepCell Tracking](https://github.com/vanvalenlab/deepcell-tracking) for creating cell lineages with deep-learning-based tracking models, and the [DeepCell Kiosk](https://github.com/vanvalenlab/kiosk-console) for deploying workflows on large datasets in the cloud. Additionally, we have developed [DeepCell Label](https://github.com/vanvalenlab/deepcell-label) for annotating high-dimensional biological images to use as training data.

Read the documentation at [deepcell.readthedocs.io](https://deepcell.readthedocs.io).

For more information on deploying models in the cloud refer to the [the Kiosk documentation](https://deepcell-kiosk.readthedocs.io).

For TensorFlow 1.X or Python 2.7 support, please use deepcell [0.7.0](https://github.com/vanvalenlab/deepcell-tf/tree/0.7.0) or earlier.

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

### Install with pip

The fastest way to get started with `deepcell-tf` is to install the package with `pip`:

```bash
pip install deepcell
```

### Install with Docker

There are also [docker containers](https://hub.docker.com/r/vanvalenlab/deepcell-tf) with GPU support available on DockerHub.
To  run the library locally on a GPU, make sure you have [CUDA](https://developer.nvidia.com/cuda-downloads) and [Docker v19.03](https://docs.docker.com/get-docker/) or later installed. For prior Docker versions, use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Alternatively, Google Cloud Platform (GCP) offers prebuilt virtual machines preinstalled with CUDA, Docker, and the NVIDIA Container Toolkit.

Once `docker` is installed, run the following command:

```bash
# Start a GPU enabled container on one GPUs
docker run --gpus '"device=0"' -it --rm \
    -p 8888:8888 \
    -v $PWD/notebooks:/notebooks \
    -v $PWD/data:/data \
    vanvalenlab/deepcell-tf:latest-gpu
```

This will start a Docker container with `deepcell-tf` installed and start a jupyter session using the default port 8888. This command also mounts a data folder (`$PWD/data`) and a notebooks folder (`$PWD/notebooks`) to the docker container so it can access data and Juyter notebooks stored on the host workstation. Data and models must be saved in these mounted directories to persist them outside of the running container. The default port can be changed to any non-reserved port by updating `-p 8888:8888` to, e.g., `-p 8080:8888`. If you run across any errors getting started, you should either refer to the `deepcell-tf` for developers section or raise an issue on GitHub.

For examples of how to train models with the `deepcell-tf` library, check out the following notebooks:

- [Training a segmentation model](https://deepcell.readthedocs.io/en/master/notebooks/Training-Segmentation.html)
- [Training a tracking model](https://deepcell.readthedocs.io/en/master/notebooks/Training-Tracking.html)

## DeepCell Applications and DeepCell Datasets

`deepcell-tf` contains two modules that greatly simplify the development and usage of deep learning models for single cell analysis. The first is <tt><a href="https://deepcell.readthedocs.io/en/master/API/deepcell.datasets.html">deepcell.datasets</a></tt>, a collection of biological images that have single-cell annotations. These data include live-cell imaging movies of fluorescent nuclei (approximately 10,000 single-cell trajectories over 30 frames), as well as static images of whole cells (both phase and fluorescence images - approximately 75,000 single cell annotations). The second is <tt><a href="https://deepcell.readthedocs.io/en/master/API/deepcell.applications.html">deepcell.applications</a></tt>, which contains pre-trained models (fluorescent nuclear and phase/fluorescent whole cell) for single-cell analysis. Provided data is scaled so that the physical size of each pixel matches that in the training dataset, these models can be used out of the box on live-cell imaging data. We are currently working to expand these modules to include data and models for tissue images. Please note that they may be spun off into their own GitHub repositories in the near future.

## DeepCell-tf for Developers

`deepcell-tf` uses `docker` and `tensorflow` to enable GPU processing. If using GCP, there are [pre-built images](https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/nvidia_gpu_cloud_image) which come with CUDA and Docker pre-installed. Otherwise, you will need to install [docker](https://docs.docker.com/install/linux/docker-ce/debian/) and [CUDA](https://developer.nvidia.com/cuda-downloads) separately.

### Build a local docker container, specifying the tensorflow version with TF_VERSION

```bash
git clone https://github.com/vanvalenlab/deepcell-tf.git
cd deepcell-tf
docker build --build-arg TF_VERSION=2.8.0-gpu -t $USER/deepcell-tf .
```

### Run the new docker image

```bash
# '"device=0"' refers to the specific GPU(s) to run DeepCell-tf on, and is not required
docker run --gpus '"device=0"' -it \
-p 8888:8888 \
$USER/deepcell-tf:latest-gpu
```

It can also be helpful to mount the local copy of the repository and the notebooks to speed up local development. However, if you are going to mount a local version of the repository, you must first run the docker image without the local repository mounted so that the C extensions can be compiled and then copied over to your local version.

```bash
# First run the docker image without mounting externally
docker run --gpus '"device=0"' -it \
-p 8888:8888 \
$USER/deepcell-tf:latest-gpu

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
    -v $PWD/notebooks:/notebooks \
    -v $PWD:/data \
    $USER/deepcell-tf:latest-gpu
```

## How to Cite

- [Deep Learning Automates the Quantitative Analysis of Individual Cells in Live-Cell Imaging Experiments](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005177)
- [Accurate cell tracking and lineage construction in live-cell imaging experiments with deep learning](https://www.biorxiv.org/content/10.1101/803205v2)
- [DeepCell Kiosk: scaling deep learning–enabled cellular image analysis with Kubernetes](https://www.nature.com/articles/s41592-020-01023-0)
- [Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning](https://www.nature.com/articles/s41587-021-01094-0)

## Copyright

Copyright © 2016-2022 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Shurl and Kay Curci Foundation, Google Research Cloud, the Paul Allen Family Foundation, & National Institutes of Health (NIH) under Grant U24CA224309-01.
All rights reserved.

## License

This software is licensed under a modified [APACHE2](https://github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE). See [LICENSE](https://github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE) for full details.

## Trademarks

All other trademarks referenced herein are the property of their respective owners.

## Credits

[![Van Valen Lab, Caltech](https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg)](http://www.vanvalen.caltech.edu/)
