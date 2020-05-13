# test.py
# Test driver for the ModelTrainer class.
#
# Right now, I'm conceiving of the ModelTrainer class as taking in the three
# conceptual elements. The elements are listed here and each element is
# followed by the named arguments that need to be passed to the ModelTrainer
# class to fully specify that element.
#
# - A dataset
# (specified by X_train, y_train, X_test, and y_test)
# - Train and validation data generators [ asumed for now to be ImageFullyConvDataGenerators]
# (specified by train_generator and validation_generator)
# - A model [ assumed for now to be bn_feature_net_skip_2D]
# (specified by model)
#
# For now, the creation of these three elements is handled in this script. They
# are each created in separate functions, though, and those functions could be
# easily moved over to the ModelTrainer class. As part of this, ModelTrainer
# would need to have its  __init__ arguments updated to include all the input
# arguments needed for creation of the element. Those decisions are all really
# about how big we want the scope of ModelTrainer to be.

import deepcell
import argparse
import json
import hashlib
import re
#import deepcell_toolbox

class ModelComparison():
    def __init__(self, metadata_file_path):
        with open(metadata_file_path) as metadata_file:
            self.source_metadata = json.load(metadata_file)

    def _compare_deepcell_version(self):
        assert self.source_metadata["deepcell_version"] == deepcell.__version__

    def _load_data(self):
        # take in dataset
        filename = 'HeLa_S3.npz'
        test_size = 0.1 # % of data saved as test
        seed = 0 # seed for random train-test split
        (self.X_train, self.y_train), (self.X_test, self.y_test) = deepcell.datasets.hela_s3.load_data(
                                                                   filename,
                                                                   test_size = test_size,
                                                                   seed = seed)

    def _compare_data(self):
        assert tuple(self.source_metadata["dataset"]["training"]["X_train_shape"]) == self.X_train.shape
        assert self.source_metadata["dataset"]["training"]["X_train_datatype"] == self.X_train.dtype.name
        assert self.source_metadata["dataset"]["training"]["X_train_md5_digest"] == hashlib.md5(self.X_train).hexdigest()
        assert tuple(self.source_metadata["dataset"]["training"]["y_train_shape"]) == self.y_train.shape
        assert self.source_metadata["dataset"]["training"]["y_train_datatype"] == self.y_train.dtype.name
        assert self.source_metadata["dataset"]["training"]["y_train_md5_digest"] == hashlib.md5(self.y_train).hexdigest()
        assert tuple(self.source_metadata["dataset"]["testing"]["X_test_shape"]) == self.X_test.shape
        assert self.source_metadata["dataset"]["testing"]["X_test_datatype"] == self.X_test.dtype.name
        assert self.source_metadata["dataset"]["testing"]["X_test_md5_digest"] == hashlib.md5(self.X_test).hexdigest()
        assert tuple(self.source_metadata["dataset"]["testing"]["y_test_shape"]) == self.y_test.shape
        assert self.source_metadata["dataset"]["testing"]["y_test_datatype"] == self.y_test.dtype.name
        assert self.source_metadata["dataset"]["testing"]["y_test_md5_digest"] == hashlib.md5(self.y_test).hexdigest()

    def _create_generators(self):
        ## training image generator
        self.train_generator = deepcell.ImageFullyConvDataGenerator()
        ## validation image generator
        self.validation_generator = deepcell.ImageFullyConvDataGenerator()
        #train_iterator = deepcell.ImageFullyConvIterator(
        #                    train_dict = train_dict,
        #                    image_data_generator = train_generator)
        #validation_iterator = deepcell.ImageFullyConvIterator(
        #                    train_dict = validation_dict,
        #                    image_data_generator = train_generator)

        ## read in parameters from metadata
        for generator_type in ("train_generator", "validation_generator"):
            if self.source_metadata["generators"][generator_type]["parameters"]["seed"]:
                self.generator_seed = self.source_metadata["generators"][generator_type]["parameters"]["seed"]
            else:
                self.generator_seed = 43
            if self.source_metadata["generators"][generator_type]["parameters"]["batch_size"]:
                self.generator_batch_size = self.source_metadata["generators"][generator_type]["parameters"]["batch_size"]
            else:
                self.generator_batch_size = 1
            if self.source_metadata["generators"][generator_type]["parameters"]["transform"]:
                self.generator_transform = self.source_metadata["generators"][generator_type]["parameters"]["transform"]
            else:
                self.generator_transform = None
            if self.source_metadata["generators"][generator_type]["parameters"]["transform_kwargs"]:
                self.generator_transform_kwargs = self.source_metadata["generators"][generator_type]["parameters"]["transform_kwargs"]
            else:
                self.generator_transform_kwargs = {}

    def _compare_generators(self):
        for generator_type in ("train_generator", "validation_generator"):
            if generator_type == "train_generator":
                data_generator_name = re.match("<([\w.]+) object at",str(self.train_generator)).group(1)
            elif generator_type == "validation_generator":
                data_generator_name = re.match("<([\w.]+) object at",str(self.validation_generator)).group(1)
            assert self.source_metadata["generators"][generator_type]["class"] == data_generator_name
            assert self.source_metadata["generators"][generator_type]["parameters"]["seed"] == self.generator_seed
            assert self.source_metadata["generators"][generator_type]["parameters"]["batch_size"] == self.generator_batch_size
            if not self.generator_transform:
                # I believe this is necessary because Python forces a None key in a JSON object into an empty list, either upon reading from or writing to disk
                assert self.source_metadata["generators"][generator_type]["parameters"]["transform"] == {}
            else:
                assert self.source_metadata["generators"][generator_type]["parameters"]["transform"] == self.generator_transform
            assert self.source_metadata["generators"][generator_type]["parameters"]["transform_kwargs"] == self.generator_transform_kwargs

    def _create_model(self):
        # create model
        receptive_field = 61
        n_skips = 3

        model_chosen = "no_skip"
        if model_chosen == "skip":
            self.model = deepcell.bn_feature_net_skip_2D(
                    n_features=22,  # segmentation mask (is_cell, is_not_cell)
                    n_skips=n_skips,
                    receptive_field=receptive_field,
                    n_conv_filters=32,
                    n_dense_filters=128,
                    input_shape=tuple(self.X_train.shape[1:]),
                    last_only=False)
        elif model_chosen == "no_skip":
            self.model = deepcell.bn_feature_net_2D(
                    n_features=22,  # segmentation mask (is_cell, is_not_cell)
                    receptive_field=receptive_field,
                    n_conv_filters=32,
                    n_dense_filters=128,
                    dilated=True,
                    input_shape=tuple(self.X_train.shape[1:]))

    def compare(self):
        # compare deepcell versions
        self._compare_deepcell_version()

        # load data and compare
        self._load_data()
        self._compare_data()

        # create and compare generators
        self._create_generators()
        self._compare_generators()

        # create model
        self._create_model()

        # train model
        self.training_kwargs = {"n_epochs": 2}
        trainer = deepcell.training.ModelTrainer(
                X_train = self.X_train,
                y_train = self.y_train,
                X_test = self.X_test,
                y_test = self.y_test,
                model = self.model,
                train_generator = self.train_generator,
                validation_generator = self.validation_generator,
                training_kwargs = self.training_kwargs)
                #generator_seed = self.generator_seed,
                #generator_batch_size = self.generator_batch_size,
                #generator_transform = self.generator_transform,
                #generator_transform_kwargs = self.generator_transform_kwargs,
        #postprocessing_fn = deepcell_toolbox.retinamask_postprocess,

        # train model
        model_name, metadata_name, model_hash = trainer.create_model()

        # compare digests for trained models
        assert self.source_metadata["model"]["trained_model_md5_digest"] == model_hash

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compare metadata saved at given location to metadata generated by new ModelTrainer instance.')
    parser.add_argument("metadata_location", help="This is the location of a previously-created ModelTrainer metadata file.")
    args = parser.parse_args()

    model_comparison = ModelComparison(args.metadata_location)
    model_comparison.compare()
    import pdb; pdb.set_trace()
