#!/usr/bin/env python
# coding: utf-8

# # 2D Nuclear Segmentation with Mask-RCNN

# In[2]:


import os
import errno

import numpy as np

import deepcell


# In[2]:


# Download the data (saves to ~/.keras/datasets)
filename = 'HeLa_S3.npz'
#(X_train, y_train), (X_test, y_test) = deepcell.datasets.hela_s3.load_data(filename)
#print('X.shape: {}\ny.shape: {}'.format(X_train.shape, y_train.shape))


# ### Set up filepath constants

# In[3]:


# the path to the data file is currently required for `train_model_()` functions

# NOTE: Change DATA_DIR if you are not using `deepcell.datasets`
# DATA_DIR = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
#
# DATA_FILE = os.path.join(DATA_DIR, filename)
#
# # confirm the data file is available
# assert os.path.isfile(DATA_FILE)


# In[4]:


# # Set up other required filepaths
#
# # If the data file is in a subdirectory, mirror it in MODEL_DIR and LOG_DIR
# PREFIX = os.path.relpath(os.path.dirname(DATA_FILE), DATA_DIR)
#
# ROOT_DIR = '/data'  # TODO: Change this! Usually a mounted volume
# MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'models', PREFIX))
# LOG_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'logs', PREFIX))
#
# # create directories if they do not exist
# for d in (MODEL_DIR, LOG_DIR):
#     try:
#         os.makedirs(d)
#     except OSError as exc:  # Guard against race condition
#         if exc.errno != errno.EEXIST:
#             raise
#

# ### Set up training parameters

# In[5]:


from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler

model_name = 'mrcnn_model_3D'
  # vgg16, vgg19, resnet50, densenet121, densenet169, densenet201

n_epoch = 3  # Number of training epochs
test_size = .10  # % of data saved as test
lr = 1e-5

optimizer = Adam(lr=lr, clipnorm=0.001)

lr_sched = rate_scheduler(lr=lr, decay=0.99)

batch_size = 8

num_classes = 1  # "object" is the only class


# ## Create the MaskRCNN Model

# In[6]:


from deepcell import model_zoo
#
# backbone = 'resnet50'
# model = model_zoo.MaskRCNN(
#     backbone=backbone,
#     input_shape=(1, 256, 256),
#     class_specific_filter=False,
#     num_classes=num_classes)

backbone = 'resnet50_3D'
model = model_zoo.MaskRCNN_3D(
    backbone=backbone,
    input_shape=(120, 256, 256, 1),
    class_specific_filter=False,
    num_classes=num_classes)

prediction_model = model


# In[7]:
#
#
# from deepcell.training import train_model_retinanet
#
# model = train_model_retinanet(
#     model=model,
#     backbone=backbone,
#     dataset=DATA_FILE,  # full path to npz file
#     model_name=model_name,
#     sigma=3.0,
#     alpha=0.25,
#     gamma=2.0,
#     include_masks=True,  # include mask generation
#     weighted_average=True,
#     score_threshold=0.01,
#     iou_threshold=0.5,
#     max_detections=100,
#     test_size=test_size,
#     optimizer=optimizer,
#     batch_size=batch_size,
#     n_epoch=n_epoch,
#     log_dir=LOG_DIR,
#     model_dir=MODEL_DIR,
#     lr_sched=lr_sched,
#     rotation_range=180,
#     flip=True,
#     shear=False,
#     zoom_range=(0.8, 1.2))
#
#
# # In[8]:
#
#
# import matplotlib.pyplot as plt
# import os
# import time
#
# import numpy as np
#
# from deepcell.utils.plot_utils import draw_detections, draw_masks
#
#
# index = np.random.randint(low=0, high=X_test.shape[0])
# print('Image Number:', index)
#
# image, mask = X_test[index:index + 1], y_test[index:index + 1]
#
# boxes, scores, labels, masks = prediction_model.predict(image)[-4:]
#
# image = 0.01 * np.tile(np.expand_dims(image[0, ..., 0], axis=-1), (1, 1, 3))
# mask = np.squeeze(mask)
#
# # copies to draw on
# draw = image.copy()
#
# # draw the masks
# draw_masks(draw, boxes[0], scores[0], masks[0],
#            score_threshold=0.5)
#
# # draw the detections
# draw_detections(draw, boxes[0], scores[0], labels[0],
#                 label_to_name=lambda x: 'cell',
#                 score_threshold=0.5,)
#
# fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 15), sharex=True, sharey=True)
# ax = axes.ravel()
#
# ax[0].imshow(image, cmap='jet')
# ax[0].set_title('Source Image')
#
# ax[1].imshow(mask, cmap='jet')
# ax[1].set_title('Labeled Data')
#
# ax[2].imshow(draw, cmap='jet')
# ax[2].set_title('Detections')
#
# fig.tight_layout()
# plt.show()


# In[ ]:




