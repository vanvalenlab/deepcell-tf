# %%
"""
Mesmer: Tissue Segmentation
===========================

Mesmer can be accessed using `deepcell.applications` with a DeepCell API key.

For more information about using a DeepCell API key, please see :doc:`/API-key`.
"""

# %%
from matplotlib import pyplot as plt

from deepcell.datasets import TissueNetSample
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay

# %%

# Download multiplex data
X, y, _ = TissueNetSample().load_data()

# %%
# create rgb overlay of image data for visualization
rgb_images = create_rgb_image(X, channel_colors=['green', 'blue'])

# %%
# plot the data
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(X[0, ..., 0], cmap='Greys_r')
ax[1].imshow(X[0, ..., 1], cmap='Greys_r')
ax[2].imshow(rgb_images[0, ...])

ax[0].set_title('Nuclear channel')
ax[1].set_title('Membrane channel')
ax[2].set_title('Overlay')

for a in ax:
    a.axis('off')

plt.show()
fig.savefig('mesmer-input.png')

# %% [markdown] raw_mimetype="text/restructuredtext"
# .. image:: ../../images/mesmer-input.png
#     :align: center
#
# The application will download pretrained weights for tissue segmentation. For more information
# about application objects, please see our
# `documentation <https://deepcell.readthedocs.io/en/master/API/deepcell.applications.html>`_.

# %%
from deepcell.applications import Mesmer
app = Mesmer()

# %% [markdown] raw_mimetype="text/restructuredtext"
# Whole Cell Segmentation
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Typically, neural networks perform best on test data that is similar to the training data.
# In the realm of biological imaging, the most common difference between datasets is the resolution
# of the data measured in microns per pixel. The training resolution of the model can be identified
# using `app.model_mpp`.

# %%
print('Training Resolution:', app.model_mpp, 'microns per pixel')

# %% [markdown] raw_mimetype="text/restructuredtext"
# The resolution of the input data can be specified in `app.predict` using the `image_mpp` option.
# The `Application` will rescale the input data to match the training resolution and then rescale
# to the original size before returning the labeled image.

# %%
segmentation_predictions = app.predict(X, image_mpp=0.5)

# %%
# create overlay of predictions
overlay_data = make_outline_overlay(rgb_data=rgb_images, predictions=segmentation_predictions)

# %%
# select index for displaying
idx = 0

# plot the data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(rgb_images[idx, ...])
ax[1].imshow(overlay_data[idx, ...])

ax[0].set_title('Raw data')
ax[1].set_title('Predictions')

for a in ax:
    a.axis('off')

plt.show()
fig.savefig('mesmer-wc.png')

# %% [markdown] raw_mimetype="text/restructuredtext"
# .. image:: ../../images/mesmer-wc.png
#     :align: center

# %% [markdown] raw_mimetype="text/restructuredtext"
# Nuclear Segmentation
# ^^^^^^^^^^^^^^^^^^^^

# %% [markdown] raw_mimetype="text/restructuredtext"
# In addition to predicting whole-cell segmentation, Mesmer can also be used for nuclear
# predictions

# %%
segmentation_predictions_nuc = app.predict(X, image_mpp=0.5, compartment='nuclear')

# %%
overlay_data_nuc = make_outline_overlay(
    rgb_data=rgb_images,
    predictions=segmentation_predictions_nuc)

# %%
# select index for displaying
idx = 0

# plot the data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(rgb_images[idx, ...])
ax[1].imshow(overlay_data_nuc[idx, ...])

ax[0].set_title('Raw data')
ax[1].set_title('Nuclear Predictions')

for a in ax:
    a.axis('off')

plt.show()
fig.savefig('mesmer-nuc.png')

# %% [markdown] raw_mimetype="text/restructuredtext"
# .. image:: ../../images/mesmer-nuc.png
#     :align: center

# %% [markdown]
# Fine-tuning the model output
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In most cases, we find that the default settings for the model work quite well across a range of
# tissues. However, if you notice specific, consistent errors in your data, there are a few things
# you can change.
#
# The first is the `interior_threshold` parameter. This controls how conservative the model is in
# estimating what is a cell vs what is background. Lower values of `interior_threshold` will
# result in larger cells, whereas higher values will result in smaller cells.
#
# The second is the `maxima_threshold` parameter. This controls what the model considers a unique
# cell. Lower values will result in more separate cells being predicted, whereas higher values
# will result in fewer cells.

# %%
# To demonstrate the effect of `interior_threshold`, we'll compare the default  with a much more
# stringent setting
segmentation_predictions_interior = app.predict(
    X,
    image_mpp=0.5,
    postprocess_kwargs_whole_cell={'interior_threshold': 0.5})
overlay_data_interior = make_outline_overlay(
    rgb_data=rgb_images,
    predictions=segmentation_predictions_interior)


# %%
# select index for displaying
idx = 0

# plot the data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(overlay_data[idx, ...])
ax[1].imshow(overlay_data_interior[idx, ...])

ax[0].set_title('Default settings')
ax[1].set_title('More restrictive interior threshold')

for a in ax:
    a.axis('off')

plt.show()
fig.savefig('mesmer-interior-threshold.png')

# %% [markdown] raw_mimetype="text/restructuredtext"
# .. image:: ../../images/mesmer-interior-threshold.png
#     :align: center

# %%
# To demonstrate the effect of `maxima_threshold`, we'll compare the default with a much more
# stringent setting
segmentation_predictions_maxima = app.predict(
    X,
    image_mpp=0.5,
    postprocess_kwargs_whole_cell={'maxima_threshold': 0.8})
overlay_data_maxima = make_outline_overlay(
    rgb_data=rgb_images,
    predictions=segmentation_predictions_maxima)


# %%
# select index for displaying
idx = 0

# plot the data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(overlay_data[idx, ...])
ax[1].imshow(overlay_data_maxima[idx, ...])

ax[0].set_title('Default settings')
ax[1].set_title('More stringent maxima threshold')

for a in ax:
    a.axis('off')

plt.show()
fig.savefig('mesmer-maxima-threshold.png')

# %% [markdown] raw_mimetype="text/restructuredtext"
# .. image:: ../../images/mesmer-maxima-threshold.png
#     :width: 400pt
#     :align: center

# %% [markdown]
# Finally, if your data doesn't include in a strong membrane marker, the model will default to just
# predicting the nuclear segmentation, even for whole-cell mode. If you'd like to add a manual
# pixel expansion after segmentation, you can do that using the `pixel_expansion` argument. This
# will universally apply an expansion after segmentation to each cell

# %%
# To demonstrate the effect of `pixel_expansion`, we'll compare the nuclear output
# with expanded output
segmentation_predictions_expansion = app.predict(
    X,
    image_mpp=0.5,
    compartment='nuclear',
    postprocess_kwargs_nuclear={'pixel_expansion': 5}
)
overlay_data_expansion = make_outline_overlay(
    rgb_data=rgb_images,
    predictions=segmentation_predictions_expansion
)


# %%
# select index for displaying
idx = 0

# plot the data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(overlay_data_nuc[idx, ...])
ax[1].imshow(overlay_data_expansion[idx, ...])

ax[0].set_title('Default nuclear segmentation')
ax[1].set_title('Nuclear segmentation with an expansion')

for a in ax:
    a.axis('off')

plt.show()
fig.savefig('mesmer-nuc-expansion.png')

# %% [markdown] raw_mimetype="text/restructuredtext"
# .. image:: ../../images/mesmer-nuc-expansion.png
#     :align: center

# %% [markdown]
# There's a separate dictionary passed to the model that controls the post-processing for
# whole-cell and nuclear predictions. You can modify them independently to fine-tune the output.
# The current defaults the model is using can be found
# `here <https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/mesmer.py#L272>`_
