# deepcell-tf
DeepCell is neural network library for single cell analysis, built using [TensorFlow](https://github.com/tensorflow/tensorflow) and [Keras](https://github.com/keras-team/keras).

## Getting Started

The fastest way to get started with DeepCell is to run the latest docker image:

```bash
nvidia-docker run -it --rm -p 8888:8888 vanvalenlab/deepcell-tf:latest
```

This will start a jupyter session, with several example notebooks detailing various training methods:

### Cell Edge and Cell Interior Segmentation

* [2D DeepCell Transform - Fully Convolutional.ipynb](scripts/deepcell/DeepCell%20Transform%202D%20Fully%20Convolutional.ipynb)

* [2D DeepCell Transform - Sample Based.ipynb](scripts/deepcell/DeepCell%20Transform%202D%20Sample%20Based.ipynb)

* [3D DeepCell Transform - Fully Convolutional.ipynb](scripts/deepcell/DeepCell%20Transfrom%203D.ipynb)

* [3D DeepCell Transform - Sample Based.ipynb](scripts/deepcell/DeepCell%20Transfrom%203D%20Sample%20Based.ipynb)

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
  -v $PWD/scripts:/deepcell-tf/scripts \
  -v /data:/data \
  $USER/deepcell-tf:latest
```
