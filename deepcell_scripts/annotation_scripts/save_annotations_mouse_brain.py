"""
save_annotations.py

Code for saving image annotations from crowdflower

@author: David Van Valen
"""

"""
Import python packages
"""

from subprocess import call
import skimage.io
import skimage.measure
import scipy.misc
import numpy as np

import warnings
import pathlib
import os
import urllib.request, urllib.parse, urllib.error
import pdb

"""
Load csv file
"""

def download_csv():
    csv_names = [
        '1257789'
        ]

    for csv_name in csv_names:
        csv_direc = os.path.join( '/data/annotation_csv', csv_name)
        csv_filename = 'f' + csv_name + '.csv'

        csv_file = os.path.join(csv_direc, csv_filename)
        if not os.path.isfile(csv_file):
            call([ "unzip", csv_file+".zip", "-d", csv_direc])
        df = pd.DataFrame.from_csv(csv_file)

        urls = df.loc[:,['annotation', 'image_url', 'broken_link']]
        for index, row in df.iterrows():
            print(row)
            
            broken_link = row['broken_link'] 

            # Get image_name
            if broken_link is False:
                annotation_url = row['annotation'][8:-2]
                image_url = row['image_url']
                
                image_folder = os.path.join(csv_direc, "nuclear", row['cell_type'], row['set_number'])
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                # generate image id
                image_url_split = image_url.split("/")
                image_id = image_url_split[-1][8:-4].zfill(4)
                # annotated image location
                annotated_image_folder = os.path.join(image_folder, "Annotation")
                if not os.path.exists(annotated_image_folder):
                    os.makedirs(annotated_image_folder)
                #pdb.set_trace()
                annotated_image_name = "annotation_" + image_id + ".tif"
                annotated_image_path = os.path.join( annotated_image_folder, annotated_image_name)
                # raw image location
                raw_image_folder = os.path.join(image_folder, "RawImages")
                if not os.path.exists(raw_image_folder):
                    os.makedirs(raw_image_folder)
                raw_image_name = "img_00000" + image_id + "_DAPI_000.jpg"
                raw_image_path = os.path.join( raw_image_folder, raw_image_name)

                #pdb.set_trace()
                print(image_url_split)
                
                # Download annotated image	
                annotated_image = urllib.request.URLopener()
                annotated_image.retrieve(annotation_url, annotated_image_path)

                # Download raw image	
                raw_image = urllib.request.URLopener()
                raw_image.retrieve(image_url, raw_image_path)

def reshape_montage(montage_file, output_folder, x_size = 256, y_size = 256, x_images = 3, y_images = 10):
    debug = False

    # open composite image
    img = scipy.misc.imread(montage_file)
    
    # create output directory
    pathlib.Path(output_folder).mkdir(exist_ok=True)
    
    # extract red channel
    img = img[:,:,0]

    # convert data to integers for convenience
    img = img.astype(np.int16)

    # chop up the montage
    x_end = x_size - 1
    y_end = y_size - 1
    images = np.ndarray( shape=(x_size, y_size, x_images*y_images), dtype=np.int16)
    image_number = 0
    while x_end < (x_size*x_images):
        # moving along columns until we get to the end of the column
        while y_end < (y_size*y_images):
            if debug:
                print("x_end: " + str(x_end))
                print("y_end: " + str(y_end))
            images[:,:,image_number] = img[ 
                    (x_end-(x_size-1)):(x_end+1), 
                    (y_end-(y_size-1)):(y_end+1) ]
            image_number += 1
            y_end += y_size
        # once we reach the end of a column, move to the beginning of the 
        # next row, and continue
        y_end = y_size - 1
        x_end += x_size

    # renumber the images so that the numbers are 1 to N
    labels = np.unique(images)
    images_copy = np.zeros(images.shape, dtype = np.int16)
    for counter, label in enumerate(labels):
        if label != 0:
            images_copy[np.where(images == label)] = counter
    images = images_copy

    # save images
    with warnings.catch_warnings():
        for image in range(images.shape[-1]):
            skimage.io.imsave(os.path.join(output_folder, str(image) + '.png'), images[:,:,image])

if __name__ == '__main__':
    montage_path = '/data/annotation_csv/1257789/nuclear/MouseBrain/set1/Annotation'
    output_path = '/data/annotation_csv/1257789/nuclear/MouseBrain/set1/Annotation_Stack'
    list_of_montages = os.listdir(montage_path)
    print(list_of_montages)

    for montage_name in list_of_montages:
        montage_file = os.path.join(montage_path, montage_name)
        subfolder = montage_name[12:-4]
        output_folder = os.path.join(output_path, subfolder)
        reshape_montage(montage_file, output_folder)
