# deepcell-tf
DeepCell is neural network API for single cell analysis, built using [TensorFlow](https://github.com/tensorflow/tensorflow) and [Keras](https://github.com/keras-team/keras).

## Getting Started

* [2D Watershed.ipynb](scripts/watershed/Watershed%20Transform%202D%20with%20FG-BG%20Separation.ipynb) A notebook for 2D instance segmentation.  It demonstrates how to load data from raw images, train a model, run that model on new data, and visualize the results.  It also compares the results of two DeepCell methods (_sample-based_ and _fully-convolutional_ semantic segmentation)

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
  $USER/deepcell-tf .
```

##### Run the docker image as a Jupyter notebook

```bash
# Alternatively, run a jupyter notebook as the container entrypoint

# Note that the --entrypoint flag is before the image name,
# but the options passed to jupyter come after the image name

NV_GPU='0' nvidia-docker run -it \
  -v $PWD/deepcell:/usr/local/lib/python3.5/dist-packages/deepcell/ \
  -v $PWD/scripts:/deepcell-tf/scripts \
  -v /data:/data \
  --entrypoint /usr/local/bin/jupyter \
  $USER/deepcell-tf:0.1 \
  notebook --allow-root --ip=0.0.0.0
```
> **_Note_**: You will need to authenticate with NGC to pull the DeepCell base image.
