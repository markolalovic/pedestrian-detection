#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" utils.py: utility functions for the base model """

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_regions(simple=False, W=512, H=256, scale=0):
    ''' Proposes regions for training and running base rcnn model. '''
    if simple:
        stride = 187
        regions = []
        for i in range(10):
            region = [5 + i*stride, 100 + (i % 2) * 20, 350, 800]
            regions.append(region)
    else:
        regions = []
        for K in [10, 20, 30]:
            width = int(W/K)
            for y in [width, 2*width, 3*width]:
                for j in range(K):
                    region = [int(j*width), y, width, int(3*width)]
                    regions.append(region)
                for j in range(K-1):
                    region = [int(j*width + width/2), y, width+20, int(3*width)]
                    regions.append(region)

    if scale > 0:
        ## resize regions
        res_regions = []
        for region in regions:
            res_region = np.array(region) / scale
            res_region = res_region.astype('int32')
            res_regions.append(res_region.tolist())
        regions = res_regions

    return regions

def get_backbone():
    ''' Backbone CNN '''
    # model_path = '/home/marko/data/models/vgg16.pt'
    # backbone = load_model(model_path)
    backbone = models.vgg16(pretrained=True)
    backbone.classifier = torch.nn.Sequential()

    for param in backbone.parameters():
        param.requires_grad = False

    return backbone

def get_IoU(bbox1, bbox2):
    x11, y11, w1, h1 = bbox1
    x21, y21, w2, h2 = bbox2

    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    x1 = max(x11, x21)
    x2 = min(x12, x22)

    y1 = max(y11, y21)
    y2 = min(y12, y22)

    width = x2 - x1
    height = y2 - y1

    if width < 0 or height < 0:
        return 0.0
    else:
        overlap = width * height

        area_a = (x12 - x11) * (y12 - y11)
        area_b = (x22 - x21) * (y22 - y21)

        combined = area_a + area_b - overlap
        return overlap / (combined + 1e-5)

def get_data(imgs_person, anno_dict, regions, W, H):
    ''' Returns:
        all_img_names, all_labels, all_diffs, all_rois, all_bboxes '''

    img_region = np.array([W, H, W, H]) # for scaling
    all_ious = []
    all_labels = []
    all_diffs = []
    all_rois = []
    all_bboxes = []
    all_img_names = []

    for i, img_name in enumerate(imgs_person):
        bboxes = anno_dict[img_name]

        bboxes = np.array([list(bbox) for bbox in bboxes])
        bboxes = bboxes.astype('float64')

        regions_ious = [[get_IoU(region, bbox) for region in regions] for bbox in bboxes]
        regions_ious = np.array(regions_ious).T

        regions_labels = []
        regions_diffs = []
        regions_rois = []

        for j, region in enumerate(regions):
            ## append label
            region_ious = regions_ious[j]
            max_idx = np.argmax(region_ious)
            max_iou = region_ious[max_idx]
            max_bbox = bboxes[max_idx]
            if max_iou > 0.17: # TODO: improve it
                regions_labels.append(1)
            else:
                regions_labels.append(0)

            ## append differences
            diff = (region - max_bbox) / img_region
            regions_diffs.append(diff)

            ## append ROI
            roi = region / img_region
            regions_rois.append(roi)

        if np.sum(regions_labels) > 0:
            all_ious.append(regions_ious)
            all_labels.append(regions_labels)
            all_diffs.append(regions_diffs)
            all_rois.append(regions_rois)
            all_bboxes.append(bboxes)
            all_img_names.append(img_name)

    ## flatten all lists
    all_labels, all_diffs, all_rois, all_bboxes = [
        item for item in [all_labels, all_diffs, all_rois, all_bboxes]]

    return all_img_names, all_labels, all_diffs, all_rois, all_bboxes

def preprocess_image(img):
    ''' Transforms image to tensor. '''

    ## first to np_array
    img = np.array(img)
    img = img / 255.
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)

    return img.to(device).float()

def decode(_y):
    _, preds = _y.max(-1)
    return preds

def save_model(model, path="./models/entire_model.pt"):
    torch.save(model, path)
    print('Model saved to ' + path)

def load_model(path="./models/entire_model.pt"):
    if torch.cuda.is_available():
        return torch.load(path)
    else:
        return torch.load(path, map_location=torch.device('cpu'))

def show_probs(probs, title=''):
    ''' Draws lollipop plot of p.m.f. of simple regions. '''
    rks = np.arange(1, 11) # or xks - regions representatives
    markerline, stemlines, baseline = plt.stem(rks, probs, markerfmt='ro')

    plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    plt.setp(stemlines, 'linestyle', 'dotted')
    plt.xticks(rks, [str(rk) for rk in rks])
    plt.title(title)
    plt.show()

def smooth_probs(probs):
    ''' Returns a 3 point running average for smoothing
    a p.m.f. of simple regions. '''
    smoothed = []
    probs = probs.tolist()
    padded = [probs[0]] + probs + [probs[-1]]
    for k in range(len(probs)):
        avg = np.mean([padded[k-1], padded[k], padded[k+1]])
        smoothed.append(avg)
    return np.array(smoothed)

def update_probs(probs, prior=[]):
    ''' Very simple Bayesian update of probs. '''
    if len(prior) == 0: # for a start to update the first frame
        prior = np.ones((len(probs),))
        prior /= sum(prior)
    posterior = probs * prior
    posterior = posterior / sum(posterior)
    return posterior
