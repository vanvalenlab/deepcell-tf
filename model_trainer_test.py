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
import deepcell_toolbox

import os
import random
import numpy as np
import tensorflow as tf

seed_value = 13 # value for all random seeds

def _load_data():
    # take in dataset
    filename = 'HeLa_S3.npz'
    test_size = 0.1 # % of data saved as test
    (X_train, y_train), (X_test, y_test) = deepcell.datasets.hela_s3.load_data(
                                                filename,
                                                test_size = test_size,
                                                seed = seed_value)
    #import pdb; pdb.set_trace()
    return (X_train, y_train), (X_test, y_test)

def _create_generators():
    ## training image generator
    train_generator = deepcell.ImageFullyConvDataGenerator()
    ## validation image generator
    validation_generator = deepcell.ImageFullyConvDataGenerator()
    #train_iterator = deepcell.ImageFullyConvIterator(
    #                    train_dict = train_dict,
    #                    image_data_generator = train_generator)
    #validation_iterator = deepcell.ImageFullyConvIterator(
    #                    train_dict = validation_dict,
    #                    image_data_generator = train_generator)
    return train_generator, validation_generator

def _create_model(X_train):
    # create model
    receptive_field = 61
    n_skips = 3
    
    model_chosen = "no_skip"
    if model_chosen == "skip":
        model = deepcell.bn_feature_net_skip_2D(
                n_features=22,  # segmentation mask (is_cell, is_not_cell)
                n_skips=n_skips,
                receptive_field=receptive_field,
                n_conv_filters=32,
                n_dense_filters=128,
                input_shape=tuple(X_train.shape[1:]),
                last_only=False)
    elif model_chosen == "no_skip":
        model = deepcell.bn_feature_net_2D(
                n_features=22,  # segmentation mask (is_cell, is_not_cell)
                receptive_field=receptive_field,
                n_conv_filters=32,
                n_dense_filters=128,
                dilated=True,
                input_shape=tuple(X_train.shape[1:]))
    return model

def _set_random_seeds():
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)

def main():
    # create model_trainer
    _set_random_seeds()
    (X_train, y_train), (X_test, y_test) = _load_data()
    train_generator, validation_generator = _create_generators()
    model = _create_model(X_train)
    training_kwargs = {"n_epochs": 2}
    trainer = deepcell.training.ModelTrainer(
            X_train = X_train,
            y_train = y_train,
            X_test = X_test,
            y_test = y_test,
            model = model,
            train_generator = train_generator,
            validation_generator = validation_generator,
            training_kwargs = training_kwargs,
            random_seed = seed_value)
    #postprocessing_fn = deepcell_toolbox.retinamask_postprocess,

    # train model
    model_name, metadata_name, model_hash = trainer.create_model()

    # Now, try to use metadata to create identical copy of model

    # Finally, compare both models

if __name__=='__main__':
    main()
