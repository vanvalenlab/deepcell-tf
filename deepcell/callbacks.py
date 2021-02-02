# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom callbacks from https://github.com/fizyer/keras-retinanet"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.keras.callbacks import Callback

from deepcell.utils.retinanet_anchor_utils import evaluate, evaluate_mask


class RedirectModel(Callback):
    """Callback which wraps another callback, but executed on a different model.

    .. code-block:: python

        model = keras.models.load_model('model.h5')
        model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
        parallel_model = multi_gpu_model(model, gpus=2)
        cb = RedirectModel(model_checkpoint, model)
        parallel_model.fit(X_train, Y_train, callbacks=[cb])

    Args:
        callback (function): callback to wrap.
        model (tf.keras.Model): model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


class Evaluate(Callback):
    """Evaluate a dataset with a model at the end of each training epoch.

    Args:
        generator (RetinaNetGenerator): The generator that represents the
            dataset to evaluate.
        iou_threshold (float): The threshold used to consider
            when a detection is positive or negative.
        score_threshold (float): The score confidence threshold to use for
            detections.
        max_detections (int): The maximum number of detections to use per image.
        save_path (str): The path to save images with visualized detections to.
        tensorboard (bool): Instance of keras.callbacks.TensorBoard used to log
            the mAP value.
        weighted_average (bool): Compute the mAP using the weighted average of
            precisions among classes.
        frames_per_batch (int): Size of z-axis. A value of 1 means 2D data.
        verbose (int): Set the verbosity level, by default this is set to 1.
    """

    def __init__(self,
                 generator,
                 iou_threshold=0.5,
                 score_threshold=0.05,
                 max_detections=100,
                 save_path=None,
                 tensorboard=None,
                 weighted_average=False,
                 frames_per_batch=1,
                 verbose=1):
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.weighted_average = weighted_average
        self.frames_per_batch = frames_per_batch
        self.verbose = verbose
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # E = evaluate_mask if self.generator.include_masks else evaluate
        E = evaluate

        # run evaluation
        avg_precisions = E(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            frames_per_batch=self.frames_per_batch,
        )

        # compute per class average precision
        instances = []
        precisions = []
        for label, (avg_precision, num_annotations) in avg_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      label,
                      'with average precision: {:.4f}'.format(avg_precision))
            instances.append(num_annotations)
            precisions.append(avg_precision)
        if self.weighted_average:
            mean_ap = sum([a * b for a, b in zip(instances, precisions)])
            mean_ap = mean_ap / sum(instances)
        else:
            mean_ap = sum(precisions) / sum(x > 0 for x in instances)

        if self.tensorboard is not None:
            import tensorflow as tf
            writer = tf.summary.create_file_writer(self.tensorboard.log_dir)
            with writer.as_default():
                tf.summary.scalar('mAP', mean_ap, step=epoch)
                writer.flush()

        logs['mAP'] = mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(mean_ap))
