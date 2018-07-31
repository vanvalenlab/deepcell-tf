"""
Custom Image loader, used for converting desired image and mask files into a format
accepted by the Retinanet MRCNN model.
"""

import os
import random
from fnmatch import fnmatch

import numpy as np
from skimage.measure import label, regionprops

import cv2

from keras_mrcnn_generator import MaskRCNNGenerator


def deepcell_data_load():
    def list_file_deepcell(direc_name, training_direcs, raw_image_direc, channel_names):
        filelist = []
        for direc in training_direcs:
            imglist = os.listdir(os.path.join(direc_name, direc, raw_image_direc))
            for channel in channel_names:
                for img in imglist:
                    # if channel string is NOT in image file name, skip it.
                    if not fnmatch(img, '*{}*'.format(channel)):
                        continue
                    image_file = os.path.join(direc_name, direc, raw_image_direc, img)
                    filelist.append(image_file)
        return sorted(filelist)

    direc_name = "/data/data/cells/HeLa/S3"
    training_direcs = ['set0']
    raw_image_direc = 'raw'
    channel_names = ['Far-red']
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
        raw_image_direc='annotated_unique',
        channel_names=['corrected'])
    print(len(train_anotedlist))

    def randomcrops(dirpaths, maskpaths, sizeX, sizeY, iteration=1):
        img = cv2.imread(dirpaths[0], 0)
        imgY = img.shape[0]
        imgX = img.shape[1]
        actX = imgX - sizeX
        actY = imgY - sizeY
        if actX < 0 or actY < 0:
            print('Image to crop is of a smaller size')
            return ([], [])
        outputi = []
        outputm = []
        while iteration > 0:
            cropindex = []
            for path in dirpaths:
                X = random.randint(0, actX)
                Y = random.randint(0, actY)
                cropindex.append((X, Y))
                image = cv2.imread(path, 0)
                newimg = image[Y:Y + sizeY, X:X + sizeX]
                newimg = np.tile(np.expand_dims(newimg, axis=-1), (1, 1, 3))
                outputi.append(newimg)

            for i, path in enumerate(maskpaths):
                image = cv2.imread(path, 0)
                X = cropindex[i][0]
                Y = cropindex[i][1]
                newimg = image[Y:Y + sizeY, X:X + sizeX]
                outputm.append(newimg)

            iteration = iteration - 1
        return (outputi, outputm)

    store = randomcrops(train_imlist, train_anotedlist, 200, 200, iteration=1)
    return store


class CSVGenerator(MaskRCNNGenerator):
    def __init__(self, base_dir=None, **kwargs):
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        self.image_stack = []

        store = deepcell_data_load()
        self.image_stack = store[0]
        # Take base_dir from annotations file if not explicitly specified.

        # parse the provided class file
        self.classes = {'cell': 0}

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = self._read_annotations(store[1])

        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(**kwargs)

    def _read_annotations(self, maskarr):
        result = {}
        for cnt, image in enumerate(maskarr):
            result[cnt] = []
            l = label(image)
            p = regionprops(l)
            cell_count = 0
            for index in range(len(np.unique(l)) - 1):
                y1, x1, y2, x2 = p[index].bbox
                result[cnt].append({
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                    'class': 'cell',
                    'mask_path':np.where(l == index + 1, 1, 0)
                })
                cell_count += 1
            print('Image number {} has {} cells'.format(cnt, cell_count))
        return result

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = self.image_stack[image_index]
        return float(image.shape[1]) / float(image.shape[0])
        # image = Image.open(self.image_path(image_index))
        # return float(image.width) / float(image.height)

    def load_image(self, image_index):
        #return read_image_bgr(self.image_path(image_index))
        #print(image_index)
        return self.image_stack[image_index]

    def load_annotations(self, image_index):
        path = self.image_names[image_index]
        annots = self.image_data[path]

        # find mask size in order to allocate the right dimension for the annotations
        annotations = np.zeros((len(annots), 5))
        masks = []

        for idx, annot in enumerate(annots):
            annotations[idx, 0] = float(annot['x1'])
            annotations[idx, 1] = float(annot['y1'])
            annotations[idx, 2] = float(annot['x2'])
            annotations[idx, 3] = float(annot['y2'])
            annotations[idx, 4] = self.name_to_label(annot['class'])
            mask = annot['mask_path']
            mask = (mask > 0).astype(np.uint8)  # convert from 0-255 to binary mask
            masks.append(np.expand_dims(mask, axis=-1))

        return annotations, masks
