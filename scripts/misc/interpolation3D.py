import numpy as np
from scipy.interpolate import RegularGridInterpolator
import deepcell
from scipy import interpolate
from skimage import io


multiplicator = 3
# Download the data (saves to ~/.keras/datasets)
filename = 'mousebrain.npz'
(X_train, y_train), (X_test, y_test) = deepcell.datasets.mousebrain.load_data(filename)
X_train_2 = X_train[:, :, :, :, 0]

io.imsave('X_train.tiff', X_train_2, 'tifffile')
print("X : ", X_train_2.shape)


#interp1d is by default linear
# data var 4D array (Batch, Z, X, Y)
def interpolation_z_axis(data, multiplicator):
    result_img = np.empty((data.shape[0], data.shape[1] * multiplicator, data.shape[2], data.shape[3]))

    for i in range(data.shape[0]):
        print(i)
        X = data[i]
        temp_image = np.empty((X.shape[0] * multiplicator, X.shape[1], X.shape[2]))
        temp_image[:] = np.nan
        for j in range(X.shape[0]):
            temp_image[j * multiplicator] = X[j]
        temp_image[-1] = temp_image[-2]
        indexes = np.arange(temp_image.shape[0])
        good = np.isfinite(temp_image).all(axis=(1, 2))
        f = interpolate.interp1d(indexes[good], temp_image[good], bounds_error=False, axis=0)
        B = f(indexes)
        result_img[i] = B

    io.imsave('InterpolatedX_train_' + str(multiplicator) + '.tiff', result_img.astype('uint8'), 'tifffile')
    print("Final shape : ", result_img.shape)
    return result_img.astype('uint8')


res = interpolation_z_axis(X_train_2, multiplicator)

