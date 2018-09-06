"""
HeLa S3 Movie Dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from tensorflow.python.keras.utils.data_utils import get_file
except:
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file

from deepcell.utils.data_utils import get_data


def load_data(path='3T3_NIH.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://deepcell-data.s3.amazonaws.com/nuclei/HeLa_S3.npz',
                    file_hash='ffe8819194fa46ee3742a2dd294074bb')

    train_dict, test_dict = get_data(path, seed=0, test_size=.2)

    x_train, y_train = train_dict['X'], train_dict['y']
    x_test, y_test = test_dict['X'], test_dict['y']
    return (x_train, y_train), (x_test, y_test)
