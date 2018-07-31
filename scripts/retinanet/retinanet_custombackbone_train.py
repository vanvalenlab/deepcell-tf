#!/usr/bin/env python

import argparse
import os
import sys
import time
import warnings
from fnmatch import fnmatch
from six import raise_from

import numpy as np

from skimage.measure import label, regionprops
from skimage.external.tifffile import TiffFile
from skimage.io import imread

from keras_retinanet import losses
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator

import tensorflow as tf
import keras

import cv2

import custom_backbone as models
from deepcell import RetinanetGenerator


def get_image(file_name):
    ext = os.path.splitext(file_name.lower())[-1]
    if ext == '.tif' or ext == '.tiff':
        img = np.asarray(np.float32(TiffFile(file_name).asarray()))
        img = np.tile(np.expand_dims(img, axis=-1), (1, 1, 3))
        return img / np.max(img)
    img = np.asarray(np.float32(imread(file_name)))
    img = np.tile(np.expand_dims(img, axis=-1), (1, 1, 3))
    return ((img / np.max(img)) * 255).astype(int)


def _parse(value, function, fmt):
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(classname):
    result = {}
    result[str(classname)] = 0
    return result


def _read_annotations(masks_list):
    result = {}
    for cnt, image in enumerate(masks_list):
        result[cnt] = []
        p = regionprops(label(image))
        cell_count = 0
        total = len(masks_list)
        for index in range(len(np.unique(label(image)))-1):
            y1, x1, y2, x2 = p[index].bbox
            result[cnt].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
            cell_count += 1
        print('-----------------Completed {} of {}-----------'.format(cnt, total))
        print('The number of cells in this image : {}'.format(cell_count))
    return result


class CSVGenerator(RetinanetGenerator):

    def __init__(self, **kwargs):

        def list_file_deepcell(direc_name,
                               training_direcs,
                               raw_image_direc,
                               channel_names):
            filelist = []
            for direc in training_direcs:
                imglist = os.listdir(os.path.join(direc_name, direc, raw_image_direc))
                #print(imglist)
                for channel in channel_names:
                    for img in imglist:
                        # if channel string is NOT in image file name, skip it.
                        if not fnmatch(img, '*{}*'.format(channel)):
                            continue
                        image_file = os.path.join(direc_name, direc, raw_image_direc, img)
                        filelist.append(image_file)
            return sorted(filelist)

        def generate_subimage(img_pathstack, HorizontalP, VerticalP, flag):
            sub_img = []
            for img_path in img_pathstack:
                img = np.asarray(np.float32(imread(img_path)))
                if flag:
                    img = (img / np.max(img))
                vway = np.zeros(VerticalP + 1)  # The dimentions of vertical cuts
                hway = np.zeros(HorizontalP + 1)  # The dimentions of horizontal cuts
                vcnt = 0  # The initial value for vertical
                hcnt = 0  # The initial value for horizontal

                for i in range(VerticalP+1):
                    vway[i] = int(vcnt)
                    vcnt += (img.shape[1] / VerticalP)

                for j in range(HorizontalP+1):
                    hway[j] = int(hcnt)
                    hcnt += (img.shape[0] / HorizontalP)

                vb = 0

                for i in range(len(hway) - 1):
                    for j in range(len(vway) - 1):
                        vb += 1

                for i in range(len(hway) - 1):
                    for j in range(len(vway) - 1):
                        sub_img.append(img[int(hway[i]):int(hway[i+1]), int(vway[j]):int(vway[j+1])])
            sub_img2 = []
            print(len(sub_img))
            if flag:
                for img in sub_img:
                    sub_img2.append(np.tile(np.expand_dims(img, axis=-1), (1, 1, 3)))
                sub_img = sub_img2

            return sub_img

        self.image_names = []
        self.image_data = {}
        self.image_stack = []
        self.mask_stack = []

        direc_name = '/data/data/cells/HeLa/S3'
        training_direcs = ['set1', 'set2']
        raw_image_direc = 'raw'
        channel_names = ['FITC']
        train_imlist = list_file_deepcell(
            direc_name=direc_name,
            training_direcs=training_direcs,
            raw_image_direc=raw_image_direc,
            channel_names=channel_names)
        print(len(train_imlist))
        print('----------------')
        train_anotedlist = list_file_deepcell(
            direc_name=direc_name,
            training_direcs=training_direcs,
            raw_image_direc='annotated',
            channel_names=['corrected'])
        print(len(train_anotedlist))

        self.image_stack = generate_subimage(train_imlist, 3, 3, True)
        self.mask_stack = generate_subimage(train_anotedlist, 3, 3, False)

        self.classes = _read_classes('cell')

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = _read_annotations(self.mask_stack)
        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """Size of the dataset."""
        return len(self.image_names)

    def num_classes(self):
        """Number of classes in the dataset."""
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """Map name to label."""
        return self.classes[name]

    def label_to_name(self, label):
        """Map label to name."""
        return self.labels[label]

    def image_path(self, image_index):
        """Returns the image path for image_index."""
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """Compute the aspect ratio for an image with image_index."""
        image = self.image_stack[image_index]
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        """Load an image at the image_index."""
        return self.image_stack[image_index]

    def load_annotations(self, image_index):
        """Load annotations for an image_index."""
        path = self.image_names[image_index]
        annots = self.image_data[path]
        boxes = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = 'cell'
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes


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
            from ..callbacks.coco import CocoEval

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
            save_best_only=True,
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
        train_generator = CSVGenerator(
            # args.annotations,
            # args.classes,
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
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

    def csv_list(string):
        return string.split(',')

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
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    if args.dataset_type == 'run':
        model = models.load_model(
            args.model_path,
            backbone_name=args.backbone,
            convert=True)
        labels_to_names = {0: 'cell'}
        makedirs(args.save_path)
        test_imlist = os.walk(args.run_path).next()[2]
        for testimgcnt, img_path in enumerate(test_imlist):
            image = get_image(img_path)
            # draw2 = np.tile(np.expand_dims(draw2,axis=-1), (1, 1, 3))
            # image = draw2
            draw2 = get_image(img_path)
            draw2 = draw2/np.max(draw2)
            # print(np.unique(image))
            # copy to draw on

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(image.shape)
            # preprocess image for network
            image = preprocess_image(image)
            # print(np.unique(image))
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
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
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
    if 'vgg' in args.backbone or 'densenet' in args.backbone or 'shvm' in args.backbone:
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
