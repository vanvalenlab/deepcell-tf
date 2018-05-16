# Use the nvidia tensorflow:18.04-py3 image as the parent image
FROM nvcr.io/vvlab/tensorflow:18.04-py3

# System maintenance
RUN apt update
RUN apt-get install -y python3-tk

# Install Retinanet
WORKDIR /deepcell-tf/lib
RUN git clone https://www.github.com/vanvalenlab/keras-retinanet
WORKDIR /deepcell-tf/lib/keras-retinanet
RUN git checkout tags/0.2
RUN pip install .

# Install Mask R-CNN
WORKDIR /deepcell-tf/lib
RUN git clone https://www.github.com/vanvalenlab/Mask_RCNN
WORKDIR /deepcell-tf/lib/Mask_RCNN
RUN pip install -r requirements.txt
RUN python setup.py install

# Set the working directory to /deepcell-tf

# Mount the contents of the deepcell-tf package and
# Install Deepcell and its requirements
WORKDIR /deepcell-tf
ADD ./deepcell_tf ./lib/deepcell_tf
WORKDIR /deepcell-tf/lib/deepcell_tf
RUN pip install -r requirements.txt
RUN python setup.py install

# Apparently, Python won't use tensorflow-gpu unless
# its version number is >= tensorflow?
RUN pip install --upgrade tensorflow-gpu

# Mount the deepcell_scripts
WORKDIR /deepcell-tf
ADD ./deepcell_scripts ./deepcell_scripts

# Change matplotlibrc file to use the Agg backend
RUN echo "backend : Agg" > /usr/local/lib/python3.5/dist-packages/matplotlib/mpl-data/matplotlibrc

# Change keras configuration file so that channels are first
#RUN mkdir $HOME/.keras
#RUN echo '{"image_data_format": "channels_first", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}' > $HOME/.keras/keras.json

# Make port 80 available to the world outside this container
EXPOSE 80
