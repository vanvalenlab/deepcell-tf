# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""Applications objects for segmentation models"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import hashlib
import numpy as np

from deepcell.utils.export_utils import export_model
from deepcell.utils.train_utils import rate_scheduler, get_callbacks
from deepcell.metrics import Metrics

from tensorflow.keras.optimizers import SGD


class ModelTrainer(object):
    def __init__(self,
                 model,
                 train_generator,
                 validation_generator,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 model_name = "test_model",
                 model_path = "test_model_folder",
                 model_version=0,
                 log_dir=None,
                 tfserving_path=None,
                 training_callbacks='default',
                 postprocessing_fn=None,
                 postprocessing_kwargs={},
                 predict_batch_size=4,
                 dataset_metadata={},
                 training_kwargs={}):

        """
        Model trainer class for segmentation models. This class eases model development by
        linking relevant metadata (dataset, training parameters, benchmarking) to the model
        training process.

        Args:
            model (tensorflow.keras.Model): The model to train.
            model_name (str):
            model_path (str):
            train_generator (tensorflow.python.keras.preprocessing.image.ImageDataGenerator):
            validation_generator (tensorflow.python.keras.preprocessing.image.ImageDataGenerator):
            log_dir (str):
            tfserving_path (str):
            training_callbacks (str):
            postprocessing_fn (function):
            postprocessing_kwargs (dict):
            predict_batch_size (int):
            model_version (int):
            dataset_metadata (dict):
            training_kwargs (dict):
        """

        # Add model information
        self.model = model
        self.model_name = model_name
        self.model_path = model_path
        self.model_version = model_version

        # Add dataset information
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Add generator information
        self.train_generator = train_generator
        self.validation_generator = validation_generator

        # Add miscellaneous information
        self.dataset_metadata = dataset_metadata
        self.postprocessing_fn = postprocessing_fn
        self.postprocessing_kwargs = postprocessing_kwargs
        self.predict_batch_size = predict_batch_size

        # Add directories for logging and model export
        if log_dir is None:
            self.log_dir = os.path.join(model_path, 'logging')
        else:
            self.log_dir = log_dir

        if tfserving_path is None:
            self.tfserving_path = os.path.join(model_path, 'serving')
        else:
            self.tfserving_path = tfserving_path

        # Add training kwargs
        self.batch_size = training_kwargs.pop('batch_size', 1)
        self.training_steps_per_epoch = training_kwargs.pop(
            'training_steps_per_epoch',
            None)
        self.validation_steps_per_epoch = training_kwargs.pop(
            'validation_steps_per_epoch',
            None)
        self.n_epochs = training_kwargs.pop('n_epochs', 8)
        self.lr = training_kwargs.pop('lr', 1e-5)
        self.lr_decay = training_kwargs.pop('lr_decay', 0.95)
        self.lr_sched = training_kwargs.pop(
            'lr_sched',
            rate_scheduler(lr=self.lr, decay=self.lr_decay))
        self.loss_function = training_kwargs.pop("loss_function",
                "mean_squared_error")
        self.optimizer = training_kwargs.pop("optimizer",
                SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))

        # Add callbacks
        if training_callbacks == 'default':
            model_name = os.path.join(model_path, model_name + '.h5')
            self.training_callbacks = get_callbacks(
                model_name, lr_sched=self.lr_sched,
                tensorboard_log_dir=self.log_dir,
                save_weights_only=False,
                monitor='val_loss', verbose=1)
            # TODO: hack. need to justify.
            del self.training_callbacks[1]
        else:
            self.training_callbacks = training_callbacks

        self.trained = False

    def _data_prep(self):
        ## parameters
        if isinstance(self.model.output_shape, list):
            skip = len(self.model.output_shape) - 1
        else:
            skip = None
        seed = 43
        batch_size = 1
        transform = None
        transform_kwargs = {}
        
        ## training image generator
        train_dict = {"X": self.X_train, "y": self.y_train}
        self.train_data = self.train_generator.flow(
            train_dict,
            skip=skip,
            seed=seed,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=transform_kwargs)

        ## validation image generator
        validation_dict = {"X": self.X_test, "y": self.y_test}
        self.validation_data = self.validation_generator.flow(
            validation_dict,
            skip=skip,
            seed=seed,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=transform_kwargs)

    def _compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

    def _train_model(self):
        if self.training_steps_per_epoch is not None:
            training_steps_per_epoch = self.training_steps_per_epoch
        else:
            training_steps_per_epoch = self.train_data.y.shape[0] // self.batch_size
        
        if self.validation_steps_per_epoch is not None:
            validation_steps_per_epoch = self.validation_steps_per_epoch
        else:
            validation_steps_per_epoch = self.validation_data.y.shape[0] // self.batch_size
        
        loss_history = self.model.fit_generator(
            self.train_data,
            steps_per_epoch=self.training_steps_per_epoch,
            epochs=self.n_epochs,
            validation_data=self.validation_data,
            validation_steps=self.validation_steps_per_epoch,
            callbacks=self.training_callbacks)

        self.trained = True
        self.loss_history = loss_history

    def _create_hash(self):
        if not self.trained:
            raise ValueError('Can only create a hash for a trained model')
        else:
            weights = []
            for layer in self.model.layers:
                weights += layer.get_weights()
            summed_weights_list = [np.sum(w) for w in weights]
            summed_weights = sum(summed_weights_list)
            model_hash = hashlib.md5(str(summed_weights).encode())
            self.model_hash = model_hash.hexdigest()

    def _benchmark(self):
        if not self.trained:
            raise ValueError('Model training is not complete')
        else:
            outputs = self.model.predict(
                self.validation_generator.x,
                batch_size=self.predict_batch_size)
            y_pred = self.postprocessing_fn(outputs, **self.postprocessing_kwargs)
            # TODO: This is a hack because the postprocessing fn returns
            # masks with no channel dimensions. This should be fixed.
            y_pred = np.expand_dims(y_pred, axis=-1)

            y_true = self.validation_generator.y.copy()
            benchmarks = Metrics(self.model_name, seg=False)
            benchmarks.calc_object_stats(y_true, y_pred)

            # Save benchmarks in dict
            self.benchmarks = {}
            for key in benchmarks.stats.keys():
                self.benchmarks[key] = int(benchmarks.stats[key].sum())

    def _create_training_metadata(self):
        training_metadata = {}
        training_metadata['batch_size'] = self.batch_size
        training_metadata['lr'] = self.lr
        training_metadata['lr_decay'] = self.lr_decay
        training_metadata['n_epochs'] = self.n_epochs
        training_metadata['training_steps_per_epoch'] = self.training_steps_per_epoch
        training_metadata['validation_steps_per_epoch'] = self.validation_steps_per_epoch

        self.training_metadata = training_metadata

    def _export_tf_serving(self):
        export_model(self.model, self.tfserving_path, model_version=self.model_version)

    def create_model(self, export_serving=False, export_lite=False):

        # Prep data generators
        self._data_prep()

        # Compile model
        self._compile_model()

        # Train model with prepped data generators
        self._train_model()

        # Load best performing weights
        model_name = os.path.join(self.model_path, self.model_name + '.h5')
        self.model.load_weights(model_name)

        # Create model hash
        self._create_hash()

        # Create benchmarks
        self._benchmark()

        # Create model metadata
        self._create_training_metadata()

        # Save model
        model_name = os.path.join(self.model_path, self.model_name + '_' + self.model_hash + '.h5')
        self.model.save(model_name)

        # Save metadata (training and dataset) and benchmarks
        metadata = {}
        metadata['model_hash'] = self.model_hash
        metadata['training_metadata'] = self.training_metadata
        metadata['dataset_metadata'] = self.dataset_metadata
        metadata['benchmarks'] = self.benchmarks

        # TODO: Saving the benchmarking object in this way saves each individual benchmark.
        # This should be refactored to save the sums.

        metadata_name = os.path.join(self.model_path, '{}_{}.json'.format(
            self.model_name, self.model_hash))

        with open(metadata_name, 'w') as json_file:
            json.dump(metadata, json_file)

        # Export tf serving model
        if export_serving:
            self._export_tf_serving()
