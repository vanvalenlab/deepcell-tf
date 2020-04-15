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

import re
import os
import json
import hashlib
import numpy as np

from deepcell import __version__
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
        print(self.postprocessing_fn)
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

        # create optimizer
        if "optimizer" in training_kwargs:
            self.optimizer_passed = True
            self.optimizer = training_kwargs.pop("optimizer")
        else:
            self.optimizer_passed = False
            self.optimizer_learning_rate = 0.01
            self.optimizer_decay = 1e-6
            self.optimizer_momentum = 0.9
            self.optimizer_nesterov = True
            self.optimizer = SGD(
                    lr= self.optimizer_learning_rate,
                    decay= self.optimizer_decay,
                    momentum= self.optimizer_momentum,
                    nesterov= self.optimizer_nesterov)

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

        self._init_output_metadata()

    def _init_output_metadata(self):
        """
        self.output_metadata should ultimately contain human-readable and hashed information
        on the following components of the training process:
            deepcell version
            dataset
            preprocessing function(s)
            image generators
            model (along with initialization state?)
            postprocessing function(s)
        """
        # initialize metadata
        self.output_metadata = {}
        # store deepcell version
        self.output_metadata["deepcell_version"] = __version__
        # store dataset information
        self.output_metadata["dataset"] = {}
        self.output_metadata["dataset"]["training"] = {}
        self.output_metadata["dataset"]["training"]["X_train_shape"] = self.X_train.shape
        self.output_metadata["dataset"]["training"]["X_train_datatype"] = self.X_train.dtype.name
        self.output_metadata["dataset"]["training"]["X_train_md5_digest"] = hashlib.md5(self.X_train).hexdigest()
        self.output_metadata["dataset"]["training"]["y_train_shape"] = self.y_train.shape
        self.output_metadata["dataset"]["training"]["y_train_datatype"] = self.y_train.dtype.name
        self.output_metadata["dataset"]["training"]["y_train_md5_digest"] = hashlib.md5(self.y_train).hexdigest()
        self.output_metadata["dataset"]["testing"] = {}
        self.output_metadata["dataset"]["testing"]["X_test_shape"] = self.X_test.shape
        self.output_metadata["dataset"]["testing"]["X_test_datatype"] = self.X_test.dtype.name
        self.output_metadata["dataset"]["testing"]["X_test_md5_digest"] = hashlib.md5(self.X_test).hexdigest()
        self.output_metadata["dataset"]["testing"]["y_test_shape"] = self.y_test.shape
        self.output_metadata["dataset"]["testing"]["y_test_datatype"] = self.y_test.dtype.name
        self.output_metadata["dataset"]["testing"]["y_test_md5_digest"] = hashlib.md5(self.y_test).hexdigest()

    def _data_prep(self):
        ## parameters
        # TODO: should be passed into ModelTrainer class
        if isinstance(self.model.output_shape, list):
            skip = len(self.model.output_shape) - 1
        else:
            skip = None
        seed = 43
        batch_size = 1
        transform = None
        transform_kwargs = {}
        
        def prep_generator(
                generator_type,
                output_metadata,
                data_generator,
                input_data_dict,
                skip = None,
                seed = None,
                batch_size = None,
                transform = None,
                transform_kwargs = {}):
            """
            Get training or validation data back from data_generator.flow() and document every step of the process
            using output_metadata.
            """


            generator_type_string = generator_type + "_generator"

            output_metadata["generators"][generator_type_string] = {}
            data_generator_name = re.match("<([\w.]+) object at",str(data_generator)).group(1)
            output_metadata["generators"][generator_type_string]["class"] = data_generator_name
            
            parameters = {}
            parameters["input_data_dict"] = {}
            for entry in input_data_dict:
                parameters["input_data_dict"][entry] = {}
                parameters["input_data_dict"][entry]["md5_digest"] = hashlib.md5(input_data_dict[entry]).hexdigest()
            parameters["skip"] = skip
            parameters["seed"] = seed
            parameters["batch_size"] = batch_size
            parameters["transform"] = {}
            try:
                transform_name = re.match("<([\w.]+) object at",str(transform)).group(1)
                parameters["transform"]["name"] = transform_name
                parameters["transform"]["md5_digest"] = hashlib.md5(transform).hexdigest()
            except AttributeError:
                pass
            parameters["transform_kwargs"] = transform_kwargs
            output_metadata["generators"][generator_type_string]["parameters"] = parameters

            output_data = data_generator.flow(
                input_data_dict,
                skip=skip,
                seed=seed,
                batch_size=batch_size,
                transform=transform,
                transform_kwargs=transform_kwargs)

            return output_data, output_metadata

        self.output_metadata["generators"] = {}
        ## training image generator
        train_dict = {"X": self.X_train, "y": self.y_train}
        self.train_data, self.output_metadata = prep_generator("train", self.output_metadata, self.train_generator, train_dict, skip, seed, batch_size, transform, transform_kwargs)
        ## validation image generator
        validation_dict = {"X": self.X_test, "y": self.y_test}
        self.validation_data, self.output_metadata = prep_generator("validation", self.output_metadata, self.validation_generator, validation_dict, skip, seed, batch_size, transform, transform_kwargs)
        print(self.output_metadata)

    def _compile_model(self):
        
        def model_compilation(
                output_metadata,
                model,
                loss_function,
                optimizer,
                optimizer_flag, 
                optimizer_learning_rate,
                optimizer_decay,
                optimizer_momentum,
                optimizer_nesterov,
                metrics):
            model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
            output_metadata["model"] = {}
            output_metadata["model"]["compilation"] = {}
            output_metadata["model"]["compilation"]["loss_function"] = loss_function
            if optimizer_flag: # optimizer was passed in by user
                output_metadata["model"]["compilation"]["optimizer"] = "need more information; can't access initialization parameters"
            else: # optimizer is set to default value
                output_metadata["model"]["compilation"]["optimizer"] = {}
                output_metadata["model"]["compilation"]["optimizer"]["class"] = re.match("<([\w.]+) object at",str(optimizer)).group(1)
                output_metadata["model"]["compilation"]["optimizer"]["learning_rate"] = optimizer_learning_rate
                output_metadata["model"]["compilation"]["optimizer"]["decay"] = optimizer_decay
                output_metadata["model"]["compilation"]["optimizer"]["momentum"] = optimizer_momentum
                output_metadata["model"]["compilation"]["optimizer"]["nesterov"] = optimizer_nesterov
            output_metadata["model"]["compilation"]["metrics"] = metrics

        metrics = ["accuracy"]
        model_compilation(
                self.output_metadata,
                self.model,
                self.loss_function,
                self.optimizer,
                self.optimizer_passed,
                self.optimizer_learning_rate,
                self.optimizer_decay,
                self.optimizer_momentum,
                self.optimizer_nesterov,
                metrics)

    def _train_model(self):
        if self.training_steps_per_epoch is not None:
            training_steps_per_epoch = self.training_steps_per_epoch
        else:
            training_steps_per_epoch = self.train_data.y.shape[0] // self.batch_size
        
        if self.validation_steps_per_epoch is not None:
            validation_steps_per_epoch = self.validation_steps_per_epoch
        else:
            validation_steps_per_epoch = self.validation_data.y.shape[0] // self.batch_size
       
        def train_model(
                output_metadata,
                model,
                train_data,
                training_steps_per_epoch,
                n_epochs,
                validation_data,
                validation_steps_per_epoch,
                training_callbacks):

            loss_history = model.fit_generator(
                train_data,
                steps_per_epoch=training_steps_per_epoch,
                epochs=n_epochs,
                validation_data=validation_data,
                validation_steps=validation_steps_per_epoch,
                callbacks=training_callbacks)
            
            output_metadata["model"]["training"] = {}
            output_metadata["model"]["training"]["n_epochs"] = n_epochs
            output_metadata["model"]["training"]["train_data"] = train_data
            output_metadata["model"]["training"]["validation_data"] = validation_data
            output_metadata["model"]["training"]["training_steps_per_epoch"] = training_steps_per_epoch
            output_metadata["model"]["training"]["validation_steps_per_epoch"] = validation_steps_per_epoch
            output_metadata["model"]["training"]["training_callbacks"] = training_callbacks


            return loss_history

        def create_hash(trained, model, output_metadata):
            if not trained:
                raise ValueError('Can only create a hash for a trained model')
            else:
                weights = []
                for layer in model.layers:
                    weights += layer.get_weights()
                summed_weights_list = [np.sum(w) for w in weights]
                summed_weights = sum(summed_weights_list)
                model_hash = hashlib.md5(str(summed_weights).encode())
                output_metadata["model"]["trained_model_md5_digest"] = model_hash.hexdigest()

        loss_history = train_model(
                self.output_metadata,
                self.model,
                self.train_data,
                self.training_steps_per_epoch,
                self.n_epochs,
                self.validation_data,
                self.validation_steps_per_epoch,
                self.training_callbacks)

        self.trained = True
        self.loss_history = loss_history

        # Load best performing weights
        model_name = os.path.join(self.model_path, self.model_name + '.h5')
        self.model.load_weights(model_name)

        create_hash(self.trained, self.model, self.output_metadata)

        print(self.output_metadata)
        import pdb; pdb.set_trace()

    def _benchmark(self):
        if not self.trained:
            raise ValueError('Model training is not complete')
        else:
            # Save benchmarks in dict
            self.benchmarks = {}
            if self.postprocessing_fn:
                outputs = self.model.predict(
                    self.validation_data.x,
                    batch_size=self.predict_batch_size)
                y_pred = self.postprocessing_fn(outputs, **self.postprocessing_kwargs)
                # TODO: This is a hack because the postprocessing fn returns
                # masks with no channel dimensions. This should be fixed.
                y_pred = np.expand_dims(y_pred, axis=-1)

                y_true = self.validation_data.y.copy()
                benchmarks = Metrics(self.model_name, seg=False)
                benchmarks.calc_object_stats(y_true, y_pred)

                # Save benchmarks
                for key in benchmarks.stats.keys():
                    self.benchmarks[key] = int(benchmarks.stats[key].sum())

    def _export_tf_serving(self):
        export_model(self.model, self.tfserving_path, model_version=self.model_version)

    def create_model(self, export_serving=False, export_lite=False):

        # Prep data generators
        self._data_prep()

        # Compile model
        self._compile_model()

        # Train model with prepped data generators
        self._train_model()

        # Create model hash
        #self._create_hash()

        # Create benchmarks
        self._benchmark()

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
            json.dump(self.output_metadata, json_file)

        # Export tf serving model
        if export_serving:
            self._export_tf_serving()

        # return information to calling program
        return model_name, metadata_name
