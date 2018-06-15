"""
settings.py

Settings file for saving shared constants
"""

from tensorflow.python.keras import backend as K

IMAGE_DATA_FORMAT = K.image_data_format()
CHANNELS_FIRST = IMAGE_DATA_FORMAT == 'channels_first'
CHANNELS_LAST = IMAGE_DATA_FORMAT == 'channels_last'
