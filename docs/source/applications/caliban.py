"""
Caliban: Nuclear Segmentation and Tracking
==========================================

TODO add api key info


"""

# %%
import copy
import os

import imageio
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from deepcell.applications import NuclearSegmentation, CellTracking
from deepcell_tracking.trk_io import load_trks
# sphinx_gallery_thumbnail_path = '../images/caliban-tracks.gif'

# %%
def shuffle_colors(ymax, cmap):
    """Utility function to generate a colormap for a labeled image"""
    cmap = mpl.colormaps[cmap].resampled(ymax)
    nmap = cmap(range(ymax))
    np.random.shuffle(nmap)
    cmap = ListedColormap(nmap)
    cmap.set_bad('black')
    return cmap


# %% [markdown]
# Prepare nuclear data
# --------------------
#
# TODO update data instructions Sample tracking data can be downloaded from https://datasets.deepcell.org/. Please adjust the path below to where your data is stored locally.

# %%
data = load_trks('/notebooks/val.trks') # Change this path
data['X'].shape

# %%
x = data['X'][0]
y = data['y'][0]

# Determine position of zero padding for removal
# Calculate position of padding based on first frame
# Assume that padding is in blocks on the edges of image
good_rows = np.where(x[0].any(axis=0))[0]
good_cols = np.where(x[0].any(axis=1))[0]
slc = (
    slice(None),
    slice(good_cols[0], good_cols[-1] + 1),
    slice(good_rows[0], good_rows[-1] + 1),
    slice(None)
)
x = x[slc]

# Determine which frames are zero padding
frames = np.sum(y, axis=(1,2))
good_frames = np.where(frames)[0] # True if image not blank
x = x[:len(good_frames)]


# %%
def plot(im):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im, 'Greys_r')
    plt.axis('off')
    plt.title('Raw Image Data')

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image

imageio.mimsave('raw.gif', [plot(x[i, ..., 0]) for i in range(x.shape[0])])

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# View .GIF of raw cells
# ^^^^^^^^^^^^^^^^^^^^^^
#
# .. image:: ../../images/raw.gif
#     :width: 300pt
#     :align: center

# %% [markdown] raw_mimetype="text/restructuredtext"
# Nuclear Segmentation
# --------------------
#
# Initialize nuclear model
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The application will download pretrained weights for nuclear segmentation. For more information about application objects, please see our [documentation](https://deepcell.readthedocs.io/en/master/API/deepcell.applications.html).

# %%
app = NuclearSegmentation()

# %% [markdown] raw_mimetype="text/restructuredtext"
# Use the application to generate labeled images
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Typically, neural networks perform best on test data that is similar to the training data. In the realm of biological imaging, the most common difference between datasets is the resolution of the data measured in microns per pixel. The training resolution of the model can be identified using `app.model_mpp`.

# %%
print('Training Resolution:', app.model_mpp, 'microns per pixel')

# %% [markdown] raw_mimetype="text/restructuredtext"
# The resolution of the input data can be specified in `app.predict` using the `image_mpp` option. The `Application` will rescale the input data to match the training resolution and then rescale to the original size before returning the labeled image.

# %%
y_pred = app.predict(x, image_mpp=0.65)

print(y_pred.shape)

# %% [markdown] raw_mimetype="text/restructuredtext"
# Save labeled images as a gif to visualize
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
ymax = np.max(y_pred)
cmap = shuffle_colors(ymax, 'tab20')

def plot(x, y):
    yy = copy.deepcopy(y)
    yy = np.ma.masked_equal(yy, 0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(x, cmap='Greys_r')
    ax[0].axis('off')
    ax[0].set_title('Raw')
    ax[1].imshow(yy, cmap=cmap, vmax=ymax)
    ax[1].set_title('Segmented')
    ax[1].axis('off')

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image

imageio.mimsave(
    './caliban-labeled.gif',
    [plot(x[i,...,0], y_pred[i,...,0])
     for i in range(y_pred.shape[0])]
)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# View .GIF of segmented cells
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `NuclearSegmentation` application was able to create a label mask for every cell in every frame!
#
# .. image:: ../../images/caliban-labeled.gif
#     :width: 500pt
#     :align: center

# %% [markdown] raw_mimetype="text/restructuredtext"
# Cell Tracking
# -------------
#
# The `NuclearSegmentation` worked well, but the cell labels of the same cell are not preserved across frames. To resolve this problem, we can use the `CellTracker`! This object will use another `CellTrackingModel` to compare all cells and determine which cells are the same across frames, as well as if a cell split into daughter cells.
#
# Initalize CellTracking application
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create an instance of `deepcell.applications.CellTracking`.

# %%
tracker = CellTracking()

# %% [markdown]
# Track the cells
# ^^^^^^^^^^^^^^^

# %%
tracked_data = tracker.track(x, y_pred)
y_tracked = tracked_data['y_tracked']

# %% [markdown] raw_mimetype="text/restructuredtext"
# Visualize tracking results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
ymax = np.max(y_tracked)
cmap = shuffle_colors(ymax, 'tab20')

def plot(x, y):
    yy = copy.deepcopy(y)
    yy = np.ma.masked_equal(yy, 0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(x, cmap='Greys_r')
    ax[0].axis('off')
    ax[0].set_title('Raw')
    ax[1].imshow(yy, cmap=cmap, vmax=ymax)
    ax[1].set_title('Tracked')
    ax[1].axis('off')

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image

imageio.mimsave(
    './caliban-tracks.gif',
    [plot(x[i,...,0], y_tracked[i,...,0])
     for i in range(y_tracked.shape[0])]
)

# %% [markdown]
# View .GIF of tracked cells
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that we've finished using `CellTracker.track_cells`, not only do the annotations preserve label across frames, but the lineage information has been saved in `CellTracker.tracks`.
#
# .. image:: ../../images/caliban-tracks.gif
#     :width: 500pt
#     :align: center
