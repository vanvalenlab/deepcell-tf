# deepcell-tf
DeepCell is neural network API for single cell analysis, built using [TensorFlow](https://github.com/tensorflow/tensorflow) and [Keras](https://github.com/keras-team/keras).

## Getting Started

There are several notebooks with various training methods that can be used as examples:

#### Cell Edge and Cell Interior Segmentation

* [2D DeepCell Transform - Fully Convolutional.ipynb](scripts/deepcell/DeepCell%20Transform%202D%20Fully%20Convolutional.ipynb)

* [2D DeepCell Transform - Sample Based.ipynb](scripts/deepcell/DeepCell%20Transform%202D%20Sample%20Based.ipynb)

* [3D DeepCell Transform - Fully Convolutional.ipynb](scripts/deepcell/DeepCell%20Transfrom%203D.ipynb)

* [3D DeepCell Transform - Sample Based.ipynb](scripts/deepcell/DeepCell%20Transfrom%203D%20Sample%20Based.ipynb)

#### Deep Watershed Instance Segmentation

* [2D Watershed - Fully Convolutional.ipynb](scripts/watershed/Watershed%20Transform%202D%20Fully%20Convolutional.ipynb)

* [2D Watershed - Sample Based.ipynb](scripts/watershed/Watershed%20Transform%202D%20Sample%20Based.ipynb)

* [3D Watershed - Fully Convolutional.ipynb](scripts/watershed/Watershed%20Transform%203D%20Fully%20Convolutional.ipynb)

* [3D Watershed - Sample Based.ipynb](scripts/watershed/Watershed%20Transform%203D%20Sample%20Based.ipynb)

## Using DeepCell with Docker

DeepCell uses `nvcr.io/nvidia/tensorflow` as a base image and `nvidia-docker` to enable GPU processing.

Below are some helpful commands to get started:

##### Build a docker image based on your local copy of deepcell-tf

```bash
# Build a docker image based on the local copy of deepcell-tf
docker build -t $USER/deepcell-tf .
```

##### Run the new docker image

```bash
# NV_GPU refers to the specific GPU to run DeepCell on, and is not required

# Mounting the codebase, scripts and data to the container is also optional
# but can be handy for local development

NV_GPU='0' nvidia-docker run -it \
  -v $PWD/deepcell:/usr/local/lib/python3.5/dist-packages/deepcell/ \
  -v $PWD/scripts:/deepcell-tf/scripts \
  -v /data:/data \
  $USER/deepcell-tf:latest
```

##### Run the docker image as a Jupyter notebook

```bash
# Alternatively, run a jupyter notebook as the container entrypoint

# Note that the --entrypoint flag is before the image name,
# but the options passed to jupyter come after the image name

NV_GPU='0' nvidia-docker run -it \
  -p 8888:8888 \
  -v $PWD/deepcell:/usr/local/lib/python3.5/dist-packages/deepcell/ \
  -v $PWD/scripts:/deepcell-tf/scripts \
  -v /data:/data \
  --entrypoint /usr/local/bin/jupyter \
  $USER/deepcell-tf:latest \
  notebook --allow-root --ip=0.0.0.0
```
> **_Note_**: You will need to authenticate with NGC to pull the DeepCell base image.
