# Use the nvidia tensorflow:18.04-py3 image as the parent image
FROM nvcr.io/vvlab/tensorflow:18.04-py3

# System maintenance
RUN apt update && apt-get install -y python3-tk
RUN pip install --upgrade pip

# Set working directory
WORKDIR /deepcell-tf

# Install Retinanet
RUN git clone https://www.github.com/vanvalenlab/keras-retinanet /deepcell-tf/lib/keras-retinanet && \
    cd lib/keras-retinanet && \
    git checkout tags/0.2 && \
    pip install . && \
    cd /deepcell-tf

# Install Mask R-CNN
RUN git clone https://www.github.com/vanvalenlab/Mask_RCNN /deepcell-tf/lib/Mask_RCNN && \
    cd /deepcell-tf/lib/Mask_RCNN && \
    pip install -r requirements.txt && \
    python setup.py install && \
    cd /deepcell-tf

# Copy the setup.py and requirements.txt and install the deepcell-tf dependencies
COPY setup.py requirements.txt /deepcell-tf/lib/deepcell-tf/
RUN pip install -r /deepcell-tf/lib/deepcell-tf/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell /deepcell-tf/lib/deepcell-tf/deepcell
COPY scripts /deepcell-tf/scripts

# Install using setup.py
RUN cd /deepcell-tf/lib/deepcell-tf/ && \
    python setup.py install

# Change matplotlibrc file to use the Agg backend
RUN echo "backend : Agg" > /usr/local/lib/python3.5/dist-packages/matplotlib/mpl-data/matplotlibrc

# Change keras configuration file so that channels are first
# RUN mkdir $HOME/.keras && \
#     echo '{"image_data_format": "channels_first", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}' > $HOME/.keras/keras.json

# Make port 80 available to the world outside this container
EXPOSE 80
