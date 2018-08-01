#!/usr/bin/env python
"""
retinanet_train.py

@author Shivam Patel

Training script adapted from keras_retinanet.
Loads images from disk instead of CSV data.
Can use custom backbones featured in deepcell_backbone.py

Usage: python retinanet_train.py \
       --no-weights --image-min-side 360 --image-max-side 426 \
       --random-transform --backbone deepcell --steps=1000 --epochs=10 \
       --gpu 0 --tensorboard-dir logs \
       csv ./annotation.csv ./classes.csv
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

from skimage.external.tifffile import TiffFile
from skimage.io import imread

import keras

from keras_retinanet import losses
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator

import tensorflow as tf

from deepcell.image_generators import RetinaNetGenerator

"""
Functions to load custom backbone from deepcell_backbone
"""

def get_backbone(backbone_name):
    """
    Returns a backbone object for the given backbone.
    """
    if 'resnet' in backbone_name:
        from keras_retinanet.models.resnet import ResNetBackbone as b
    elif 'mobilenet' in backbone_name:
        from keras_retinanet.models.mobilenet import MobileNetBackbone as b
    elif 'vgg' in backbone_name:
        from keras_retinanet.models.vgg import VGGBackbone as b
    elif 'densenet' in backbone_name:
        from keras_retinanet.models.densenet import DenseNetBackbone as b
    elif 'deepcell' in backbone_name:
        # Import custom written backbone class here
        from deepcell_backbone import DeepcellBackbone as b
    else:
        raise NotImplementedError('Backbone class for `{}` not implemented.'.format(
            backbone_name))

    return b(backbone_name)


def load_model(filepath,
               backbone_name='resnet50',
               convert=False,
               nms=True,
               class_specific_filter=True):
    """Loads a retinanet model using the correct custom objects.
    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name: Backbone with which the model was trained.
        convert: Boolean, whether to convert the model to an inference model.
        nms: Boolean, whether to add NMS filtering to the converted model.
             Only valid if convert=True.
        class_specific_filter: Whether to use class specific filtering or filter
                               for the best scoring class only.
    # Returns
        A keras.models.Model object.
    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    from keras.models import load_model as keras_load_model

    model = keras_load_model(filepath, custom_objects=get_backbone(backbone_name).custom_objects)
    if convert:
        model = retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter)

    return model


"""
Functions to train retinanet models
"""


def get_image(file_name):
    ext = os.path.splitext(file_name.lower())[-1]
    if ext in {'.tif', '.tiff'}:
        img = np.asarray(np.float32(TiffFile(file_name).asarray()))
        img = np.tile(np.expand_dims(img, axis=-1), (1, 1, 3))
        return img / np.max(img)
    img = np.asarray(np.float32(imread(file_name)))
    img = np.tile(np.expand_dims(img, axis=-1), (1, 1, 3))
    return ((img / np.max(img)) * 255).astype(int)


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """Construct a modified tf session."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """Load weights for model.
    # Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False):
    """Creates three models (model, training_model, prediction_model).
    # Args
        backbone_retinanet: A function to call to create a retinanet model with a given backbone.
        num_classes: The number of classes to train.
        weights: The weights to load into the model.
        multi_gpu: The number of GPUs to use for training.
        freeze_backbone: If True, disables learning for the backbone.
    # Returns
        model: The base model. This is also the model that is saved in snapshots.
        training_model: The training model. If multi_gpu=0, this is identical to model.
        prediction_model: The model wrapped with utility functions to perform object detection
                           (applies regression values and performs NMS).
    """
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU
    # to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(
                backbone_retinanet(num_classes, modifier=modifier),
                weights=weights,
                skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(
            backbone_retinanet(num_classes, modifier=modifier),
            weights=weights,
            skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """Creates the callbacks to use during training.
    # Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
    # Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None)
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from keras_retinanet.callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(
                    backbone=args.backbone,
                    dataset_type=args.dataset_type)
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks


def create_generators(args, preprocess_image):
    """Create generators for training and validation.
    # Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'image_min_side': args.image_min_side,
        'image_max_side': args.image_max_side,
        'preprocess_image': preprocess_image,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if args.dataset_type == 'train':
        train_generator = RetinaNetGenerator(
            direc_name='/data/data/cells/HeLa/S3',
            training_dirs=['set1', 'set2'],
            raw_image_dir='raw',
            channel_names=['FITC'],
            annotation_dir='annotated',
            annotation_names=['corrected'],
            # args.annotations,
            # args.classes,
            **common_args
        )

        if args.val_annotations:
            validation_generator = RetinaNetGenerator(
                direc_name='/data/data/cells/HeLa/S3',
                training_dirs=['set1', 'set2'],
                raw_image_dir='raw',
                channel_names=['FITC'],
                annotation_dir='annotated',
                annotation_names=['corrected'],
                # args.val_annotations,
                # args.classes,
                **common_args
            )
        else:
            validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
    # Args
        parsed_args: parser.parse_args()
    # Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError('Batch size ({}) must be equal to or higher than '
                         'the number of GPUs ({})'.format(
                             parsed_args.batch_size,
                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError('Multi GPU training ({}) and resuming from snapshots '
                         '({}) is not supported.'.format(
                             parsed_args.multi_gpu,
                             parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError('Multi-GPU support is experimental, use at own risk! '
                         'Run with --multi-gpu-force if you wish to continue.')

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been '
                      'properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('train')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    csv_parser_run = subparsers.add_parser('run')
    csv_parser_run.add_argument('run_path', help='Path to folder containing test data')
    csv_parser_run.add_argument('model_path', help='Path to the model(.h5) file')
    csv_parser_run.add_argument('--save_path', help='Path to save data', default='./test_output')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights', help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=False)
    group.add_argument('--weights', help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False, default=True)

    parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=10)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=1000)
    parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create object that stores backbone information
    backbone = get_backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    if args.dataset_type == 'run':
        model = load_model(
            args.model_path,
            backbone_name=args.backbone,
            convert=True)
        labels_to_names = {0: 'cell'}
        makedirs(args.save_path)
        test_imlist = next(os.walk(args.run_path))[2]
        for testimgcnt, img_path in enumerate(test_imlist):
            image = get_image(img_path)
            draw2 = get_image(img_path)
            draw2 = draw2 / np.max(draw2)
            # copy to draw on

            # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image)
            print(scale)
            # process image
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            print('processing time: ', time.time() - start)

            # correct for image scale
            boxes /= scale

            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break

                color = label_color(label)
                color = [255, 0, 255]
                b = box.astype(int)
                draw_box(draw2, b, color=color)

                caption = '{} {:.3f}'.format(labels_to_names[label], score)
                draw_caption(draw2, b, caption)
            plt.imsave(os.path.join(args.save_path, 'retinanet_output_' + str(testimgcnt)), draw2)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        prediction_model = retinanet_bbox(model=model)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone
        )

    print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if any(x in args.backbone for x in ['vgg', 'densenet', 'deepcell']):
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
