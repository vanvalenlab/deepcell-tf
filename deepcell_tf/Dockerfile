# Use the nvidia tensorflow:17.12 image as the parent image
FROM nvcr.io/vvlab/tensorflow:17.12

# Set the working directory to /deepcell-tf
WORKDIR /deepcell-tf

# Add the contents of the working directory to /deepcell-tf
ADD . /deepcell-tf

# Install the packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install the deep cell package
RUN python setup.py install

# Change matplotlibrc file to use the Agg backend
RUN echo "backend : Agg" > /usr/local/lib/python2.7/dist-packages/matplotlib/mpl-data/matplotlibrc

# Change keras configuration file so that channels are first
RUN mkdir $HOME/.keras
RUN echo '{"image_data_format": "channels_first", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}' > $HOME/.keras/keras.json 

# Make port 80 available to the world outside this container
EXPOSE 80


