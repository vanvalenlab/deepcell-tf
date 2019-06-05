# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=nightly-gpu
FROM tensorflow/tensorflow:${TF_VERSION}-py3-jupyter

RUN mkdir /tf/notebooks

# System maintenance
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    python3-tk \
    libsm6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/pip install --upgrade pip

# Copy the setup.py and requirements.txt and install the deepcell-tf dependencies
COPY setup.py requirements.txt /opt/deepcell-tf/
RUN pip install -r /opt/deepcell-tf/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell /opt/deepcell-tf/deepcell

# Install deepcell via setup.py
RUN pip install /opt/deepcell-tf && \
    cd /opt/deepcell-tf && \
    python setup.py build_ext --inplace

# Copy over deepcell notebooks
COPY scripts/ /tf/notebooks/

# Change matplotlibrc file to use the Agg backend
RUN echo "backend : Agg" > /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc


