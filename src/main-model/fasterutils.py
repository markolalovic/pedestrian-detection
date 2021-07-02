#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" fasterutils.py: utility functions for the faster rcnn model """

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
import transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model():
    ''' Loads faster R-CNN model with ResNet50 backbone pre-trained on COCO dataset.

    We use the backbone as a feature extractor only. For this we:

    * call fasterrcnn_resnet50_fpn API with:
        pretrained_backbone = True
    * set backbone layers parameters to:
        param.requires_grad = False

    We also use pretrained:
        * roi_heads (region of interest pooling layers)
        * rpn.head (classification and regression heads layers)
    but fine tune their parameters on Cityscapes dataset using annotations from
    Citypersons dataset.
    '''

    model = models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, # pretrained rpn.head and roi_heads
        pretrained_backbone=True # pretrained backbone
    )

    ## replace the classifier with a new one, that has
    ## num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background

    ## get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    ## replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    ## freeze the backbone layers
    for param_name, param in model.named_parameters():
        if 'backbone' in param_name:
            param.requires_grad = False

    # we will fine tune rpn.head and roi_heads during training

    return model

def get_transform(train):
    ''' Converts the image, a PIL image, into a PyTorch Tensor.
    No need to add a mean and std normalization nor image rescaling in the data
    transforms as those are handled internally by the Faster R-CNN model.'''

    transforms = []
    transforms.append(T.ToTensor())

    ## data augmentation:
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

        ## TODO: add Highlight augmentation and compare results to see the effect

    return T.Compose(transforms)

def save_model(model, path="./models/entire_model.pt"):
    torch.save(model, path)
    print('Model saved to ' + path)
    
def load_model(path="./models/entire_model.pt"):
    if torch.cuda.is_available():
        return torch.load(path)
    else:
        return torch.load(path, map_location=torch.device('cpu'))






#@
