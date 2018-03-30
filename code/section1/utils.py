#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:09:04 2018

Mastering PyTorch for Deep Learning

@author: pbialecki
"""

import re
import os
import errno

from sklearn.model_selection import train_test_split

SEED=2809


def get_image_id(image_name):
    '''
    Returns the image id regardless of the smoothing factor.
    '''
    pattern = '_\w{1}(\d+_C\d+)_F\d+_(s\d+)'
    match = re.findall(pattern, image_name)
    image_id = '_'.join(match[0])
    return image_id


def get_image_name(image_path):
    '''
    Returns the image name given the path
    '''
    image_name = image_path.split('/')[-1].split('.')[0]
    return image_name

def get_number_of_cells(image_name):
    '''
    Returns the number of cells for the current image.
    '''
    pattern = '\w+_\w+\d+_C(\d+)_'
    nb_cells = int(re.findall(pattern, image_name)[0])
    return nb_cells

def split_data(image_paths, target_paths):
    '''
    Splits the data into a training and a validation set.
    '''
    nb_cells = [get_number_of_cells(im_path) for im_path in image_paths]
    im_path_train, im_path_val, tar_path_train, tar_path_val = train_test_split(
        image_paths,
        target_paths,
        test_size=0.1,
        random_state=SEED,
        stratify=nb_cells)

    return im_path_train, im_path_val, tar_path_train, tar_path_val

def download_data(root='./'):
    '''
    Downloads the BBBC005 dataset from: 
    https://data.broadinstitute.org/bbbc/BBBC005/
    '''
    from six.moves import urllib
    import zipfile

    folder = os.path.expanduser('data')
    data_url = 'https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip'
    target_url = 'https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip'

    data_folder = data_url.split('/')[-1].replace('.zip', '')
    target_folder = target_url.split('/')[-1].replace('.zip', '')

    if os.path.exists(os.path.join(root, folder)) and \
        os.path.exists(os.path.join(root, folder, 'data_paths.txt')):
        return

    # Download dataset if it doesn't exist already
    try:
        os.makedirs(os.path.join(root, folder))
        os.makedirs(os.path.join(root, folder, data_folder))
        os.makedirs(os.path.join(root, folder, target_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    print('Downloading ' + data_url)
    data = urllib.request.urlopen(data_url)
    filename = data_url.rpartition('/')[2]
    file_path = os.path.join(root, folder, filename)
    with open(file_path, 'wb') as f:
        f.write(data.read())
    with zipfile.ZipFile(file_path, 'r') as zip_f:
        zip_f.extractall(os.path.join(root, folder))
    os.unlink(file_path)

    print('Downloading ' + target_url)
    data = urllib.request.urlopen(target_url)
    filename = target_url.rpartition('/')[2]
    file_path = os.path.join(root, folder, filename)
    with open(file_path, 'wb') as f:
        f.write(data.read())
    with zipfile.ZipFile(file_path, 'r') as zip_f:
        for name in zip_f.namelist():
            if name.startswith('BBBC005_v1_ground_truth/'):
                zip_f.extract(name, os.path.join(root, folder))
    os.unlink(file_path)



