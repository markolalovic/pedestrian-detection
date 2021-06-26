#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" data-baseline.py: Prepares a dataset for baseline model:

* images of single person with large bounding box areas (../data/single-person dir)
* random set of images without people (../data/no-person dir)

TODO: figure out the ratio of person to no-person images.
"""

import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from shutil import copyfile
import random

imgs_path = '../datasets/cityscapes-dataset/leftImg8bit/train/'
anno_path = '../datasets/CityPersons/annotations/'

imgs1_path = '../datasets/baseline-dataset/person'
imgs0_path = '../datasets/baseline-dataset/noperson'

def get_annotations(anno_path):
    ''' Prepares data - only train data for now, by
    transforming annotations from .mat format to a dictionary.

    Changed it so we take annotations of all people and not only
    pedestrians as before.
    '''

    anno_train = scipy.io.loadmat(anno_path + 'anno_train.mat')
    anno_train = anno_train['anno_train_aligned']

    d = {}
    for i in range(anno_train[0].shape[0]):
        # extract data from the annotations matrix
        city_name = anno_train[0, i][0][0][0][0]
        img_name = anno_train[0, i][0][0][1][0]

        bboxes = []
        for bb in anno_train[0, i][0][0][2]:
            ## format is: [class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis]
            if bb[0] > 0:
                #class_label =1: pedestrians
                #class_label =2: riders (e.g. cyclist)
                #class_label =3: sitting persons
                #class_label =4: other persons with unusual postures
                #class_label =5: group of people
                bboxes.append(bb[1:5]) # bbox = [x, y, w, h]

        d[img_name] = bboxes

    return d


if __name__ == "__main__":
    anno_dict = get_annotations(anno_path)
    print('There are %d examples' % len(anno_dict))

    imgs0 = [] # images without people
    imgs1 = [] # images with exactly one person
    for img_name in anno_dict.keys():
        n = len(anno_dict[img_name])
        if n == 0:
            imgs0.append(img_name)
        elif n == 1:
            imgs1.append(img_name)

    print('No people images: %d, one person images: %d.'
        % (len(imgs0), len(imgs1)))

    ## rank them by areas in descending order
    areas = []
    for img_name in imgs1:
        bboxes = anno_dict[img_name]
        area = int(bboxes[0][2]) * int(bboxes[0][3])
        areas.append(area)

    ranked = np.argsort(areas)
    ranked = ranked[::-1]

    ## move top 100 single person images to
    found = 0
    for i in range(100):
        img_name = imgs1[ranked[i]]
        city_name = img_name.split('_')[0]
        img_path = imgs_path + city_name + '/' + img_name

        source = img_path
        destination = imgs1_path + img_name

        if os.path.isfile(source):
            found+=1
        else:
            print(i)

        copyfile(source, destination)
    print('Moved %d single person images.' %(found))

    ## move 100 random images without people to
    found = 0
    for img_name in random.sample(imgs0, k=100):
        city_name = img_name.split('_')[0]
        img_path = imgs_path + city_name + '/' + img_name

        source = img_path
        destination = imgs0_path + img_name

        if os.path.isfile(source):
            found+=1
        else:
            print(i)

        copyfile(source, destination)
    print('Moved %d images without people.' %(found))
