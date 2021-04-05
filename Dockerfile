# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=2.4.1-gpu

FROM tensorflow/tensorflow:${TF_VERSION}

# System maintenance
# System maintenance
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-tk \
    graphviz \
    libxext6 \
    libxrender-dev \
    libsm6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/bin/python3 -m pip install --upgrade pip

#WORKDIR notebooks/

# Copy the required setup files and install the deepcell-tf dependencies
COPY setup.py README.md requirements.txt /opt/deepcell-tf/

# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow>/d" /opt/deepcell-tf/requirements.txt && \
    pip install -r /opt/deepcell-tf/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell /opt/deepcell-tf/deepcell

# Install deepcell via setup.py
RUN pip install /opt/deepcell-tf

## need to remove jedi - causes erros for tab in jupyter
RUN pip uninstall jedi --yes

# Copy over deepcell notebooks
COPY notebooks/ /notebooks/

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
