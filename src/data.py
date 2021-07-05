#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" data.py: A custom dataset for CityPersons. """

import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class HamburgDataset(torch.utils.data.Dataset):
    '''
    Prepares Torch.utils.data.Dataset class for citypersons dataset.
    It only works with images that have pedestrians.
    TODO: If necessary, we need to customize it so we can create a test dataset
    with images that don't have pedestrians.
    '''

    def __init__(self, root, anno_dict, transforms=None):
        '''
        root: img_path (e.g. hamburg directory of images)
        anno_dict:
        '''
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root)))
        self.anno = anno_dict

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')

        ## prepare bboxes coordinates
        # transform from [x, y, w, h] to [x0, y0, x1, y1]
        boxes = []
        for bbox in self.anno[self.imgs[idx]]:
            x, y = bbox[0], bbox[1]
            w, h = bbox[2], bbox[3]
            boxes.append([x, y, x+w, y+h])

        # transform to torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        ## define labels, there is only one class
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        ## other definitions
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


## some helper functions

#returns annotation for one .txt file in (x,y,width,height)
def get_annotation_caltech(anno_path):
    myfile = open(anno_path, "rt") # open lorem.txt for reading text
    contents = myfile.read()
    #print(contents)
    bboxesTemp = contents.splitlines()
    #print(bboxesTemp)
    bboxes = []
    print(bboxesTemp)
    for x in range(len(bboxesTemp)):
        bbox = [0,0,0,0]
        Placeholder = bboxesTemp[x]
        Placeholder = Placeholder.split()
        bbox[0] = int(640*float(Placeholder[1])-640*float(Placeholder[3])*0.5)
        bbox[1] = int(480*float(Placeholder[2])-480*float(Placeholder[4])*0.5)
        bbox[2] = int(640*float(Placeholder[3]))
        bbox[3] = int(480*float(Placeholder[4]))
        bboxes.append(bbox)
        

    return bboxes
    
#returns a dictionary with the Images as Keys and annotations in (x,y,width,height)
def get_annotations_caltech():
    dict = {}
    files = [f for f in listdir("./datasets/caltech/CaltechImages00") if isfile(join("./datasets/caltech/CaltechImages00", f))]
    files.remove(".DS_Store")
    for f in files:

        dict[f] = get_annotation_caltech("./datasets/caltech/CaltechAnnots/train" + f[3:(len(f)-4)] +".txt")
    return dict


def get_annotations(anno_path, anno_val=False):
    ''' Returns annotations as a dictionary from .mat format. '''
    
    if anno_val:
        anno_train = scipy.io.loadmat(anno_path + 'anno_val.mat')
        anno_train = anno_train['anno_val_aligned']        
    else:
        anno_train = scipy.io.loadmat(anno_path + 'anno_train.mat')
        anno_train = anno_train['anno_train_aligned']

    d = {}
    for i in range(anno_train[0].shape[0]):
        # extract data from the annotations matrix
        city_name = anno_train[0, i][0][0][0][0]
        img_name = anno_train[0, i][0][0][1][0]

        bboxes = []
        for bb in anno_train[0, i][0][0][2]:
            if bb[0] > 0: # class_label = 1 means it is a person
                bboxes.append(bb[1:5]) # bbox format = [x, y, w, h]
        
        ## keep only images with persons
        if bboxes != []:
            d[img_name] = bboxes
    
    return d

def show(img_path, img_name, anno_dict):
    '''Shows the image and corresponding annotations.'''

    img = Image.open(img_path + img_name)
    bboxes = anno_dict[img_name]

    plt.rcParams['figure.figsize'] = [12, 8]

    fig, ax = plt.subplots()
    ax.imshow(img);

    # bbox = [x, y, w, h]
    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    plt.title(img_name)
    plt.show()


## testing on images from Hambrug
if __name__ == "__main__":
    img_path = './datasets/citypersons/hamburg/'
    anno_path = './datasets/citypersons/CityPersons/annotations/'

    anno_dict = get_annotations(anno_path)
    print('There are %d examples from Hamburg' % len(anno_dict))
    print('\nFirst example: ')

    img_name = list(anno_dict.keys())[0]
    print(img_name)
    print('\nAnnotations: ')
    anno_dict[img_name]

    imgs = list(sorted(os.listdir(img_path)))
    print(imgs[:10])

    # let's check the first image
    show(img_path, imgs[0], anno_dict)

    ## we only keep the images with pedestrians
    for img in imgs:
        if anno_dict[img] == []:
            os.remove(img_path + img)
    imgs = list(sorted(os.listdir(img_path)))
    print(len(imgs))

    dataset = HamburgDataset(img_path, anno_dict)

    # show the data of the first image:
    print(dataset[0])
