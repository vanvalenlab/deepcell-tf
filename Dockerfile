# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=1.11.0

FROM tensorflow/tensorflow:${TF_VERSION}-py3

# System maintenance
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-tk \
    libsm6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/pip install --upgrade pip

WORKDIR /notebooks

# Copy the setup.py and requirements.txt and install the deepcell-tf dependencies
COPY setup.py requirements.txt /opt/deepcell-tf/
RUN pip install -r /opt/deepcell-tf/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell /opt/deepcell-tf/deepcell

# Install deepcell via setup.py
RUN pip install /opt/deepcell-tf && \
    cd /opt/deepcell-tf && \
    python setup.py build_ext --inplace

# Older versions of TensorFlow have notebooks, but they may not exist
RUN if [ -n "$(find /notebooks/ -prune)" ] ; then \
      mkdir -p /notebooks/intro_to_tensorflow && \
      ls -d /notebooks/* | grep -v intro_to_tensorflow | \
      xargs -r mv -t /notebooks/intro_to_tensorflow ; \
    fi

# Copy over deepcell notebooks
COPY scripts/ /notebooks/

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
