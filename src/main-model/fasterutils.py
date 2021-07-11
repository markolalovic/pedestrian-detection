#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" fasterutils.py: utility functions for the faster rcnn model """

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import torchvision.transforms as transforms
## imports for how-to:
## import transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def get_model_anchors(
    sizes=tuple([(4, 8, 16, 32, 64) for _ in range(5)]),
    aspect_ratios=tuple([(0.5, 1.0, 2.3) for _ in range(5)])):
    ''' Same as get_model() but with redefined region proposal network.

    Main parameter values are taken from Pedestron (Elephant in the room paper):
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],

    Other parameters to set:
    'fg_iou_thresh', 'bg_iou_thresh', 'batch_size_per_image', 'positive_fraction',
    'pre_nms_top_n', 'post_nms_top_n', and 'nms_thresh'

    References:

        * Pedestrons faster_rcnn_r50_fpn config:

            * https://github.com/hasanirtiza/Pedestron/blob/master/configs/faster_rcnn_r50_fpn_1x.py

        * How to change parameters in torchvision fasterrcnn implementation:

            * https://github.com/pytorch/vision/blob/master/torchvision/models/detection/rpn.py
            * https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
            * https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    '''

    model = models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, # pretrained rpn.head and roi_heads
        pretrained_backbone=True # pretrained backbone
    )

    ## only person class + background class
    num_classes = 2

    ## get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    ## replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    ## define the region proposal network
    anchor_generator = AnchorGenerator(sizes, aspect_ratios)
    model.rpn = RegionProposalNetwork(
        anchor_generator = anchor_generator,
        head = RPNHead(256, anchor_generator.num_anchors_per_location()[0]),
        fg_iou_thresh = 0.5, #.7
        bg_iou_thresh = 0.5, #.3
        batch_size_per_image = 48, # use fewer proposals
        positive_fraction = 0.5,
        pre_nms_top_n = dict(training=200, testing=100),
        post_nms_top_n = dict(training=160, testing=80),
        nms_thresh = 0.7)

    ## freeze the backbone layers, and fine tune rpn.head and roi_heads
    for param_name, param in model.named_parameters():
        if 'backbone' in param_name:
            param.requires_grad = False

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

def show_losses(all_mean_losses):
    ''' Plots train losses given all_mean_losses from running train_one_epoch() for all epochs. '''

    losses_sum = np.array([[np.sum(mean_losses) for mean_losses in all_mean_losses]])
    losses_all = np.array(all_mean_losses)
    losses_values = np.hstack((losses_sum.T, losses_all))
    losses_names = ['loss_sum', 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']

    for i in range(4):
        lab = losses_names[i]
        x = np.arange(len(losses_values[:, i]))
        y = losses_values[:, i]
        plt.plot(x, y, label=lab)

    plt.legend(loc='best')
    plt.show()

def show(img, bboxes_x0x1y0y1, scores, threshold=0.7):
    ''' For the How-To. '''

    plt.rcParams['figure.figsize'] = [12, 8]

    fig, ax = plt.subplots()
    ax.imshow(img);

    bboxes = []
    for i, bbox in enumerate(bboxes_x0x1y0y1):
        if scores[i] > threshold:
            bbox = list(bbox)
            x0, y0 = bbox[0], bbox[1]
            x1, y1 = bbox[2], bbox[3]
            bboxes.append([x0, y0, x1 - x0, y1 - y0])

    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    plt.title('Example from Hamburg')
    plt.show()
