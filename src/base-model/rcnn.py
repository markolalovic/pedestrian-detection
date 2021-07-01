#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" rcnn.py: base model """

import numpy as np
from PIL import Image
import torchvision.models as models
import torch
import torch.nn as nn

from utils import preprocess_image
from utils import decode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset():
    def __init__(self, img_paths, rois, labels, diffs, bboxes):
        self.img_paths = img_paths
        self.bboxes = bboxes
        self.rois = rois
        self.labels = labels
        self.diffs = diffs

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = str(self.img_paths[i])
        image = Image.open(img_path)
        np_image = np.array(image)
        H, W = np_image.shape[:2]

        bboxes = self.bboxes[i]
        rois = self.rois[i]

        img_region = np.array([W, H, W, H])
        roi_bboxes = np.array(rois) * img_region
        roi_bboxes = roi_bboxes.astype(np.uint16)

        x0y0x1y1s = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in roi_bboxes]
        crops = [np_image[y0:y1, x0:x1] for (x0, y0, x1, y1) in x0y0x1y1s]
        ## transform crops to images
        crops = [Image.fromarray(crop, 'RGB') for crop in crops]

        labels = self.labels[i]
        diffs = self.diffs[i]

        return image, crops, roi_bboxes, labels, diffs, bboxes, img_path

    def collate_fn(self, batch):
        ''' Resizes and normalizes crop image. '''

        input, rois, rixs, labels, diffs = [], [], [], [], []
        for i in range(len(batch)):
            img, crops, roi_bboxes, img_labels, img_diffs, bboxes, img_paths = batch[i]

            newsize = (224, 224)
            crops = [crop.resize(newsize) for crop in crops]
            crops = [preprocess_image(crop)[None] for crop in crops]
            input.extend(crops)

            labels.extend([c for c in img_labels])
            diffs.extend(img_diffs)

        input = torch.cat(input).to(device)
        labels = torch.Tensor(labels).long().to(device)
        diffs = torch.Tensor(diffs).float().to(device)

        return input, labels, diffs

class BaseModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        feature_dim = 25088
        self.backbone = backbone
        self.cls_score = nn.Linear(feature_dim, 2) # TODO: 2 classes
        self.bbox = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh())
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()

    def forward(self, input):
        features = self.backbone(input) # extact features
        score = self.cls_score(features) # get pedestrian likelihood
        bbox = self.bbox(features) # get regressed bbox

        return score, bbox

    def calc_loss(self, probs, _diffs, labels, diffs):
        d_loss = self.cel(probs, labels)
        indices, = torch.where(labels == 1)
        _diffs = diffs[indices]
        diffs = diffs[indices]
        self.lmb = 10.0

        if len(indices) > 0:
            r_loss = self.sl1(_diffs, diffs)

            return d_loss + self.lmb * r_loss, d_loss.detach(), r_loss.detach()
        else:
            r_loss = 0
            return d_loss + self.lmb * r_loss, d_loss.detach(), r_loss

def train_batch(inputs, model, optimizer, criterion):
    input, classes, diffs = inputs
    model.train()
    optimizer.zero_grad()
    _classes, _diffs = model(input)
    loss, loc_loss, regr_loss = criterion(_classes, _diffs, classes, diffs)
    accs = classes == decode(_classes)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

@torch.no_grad()
def validate_batch(inputs, model, criterion):
    input, classes, diffs = inputs
    with torch.no_grad():
        model.eval()
        _classes, _diffs = model(input)
        loss, loc_loss, regr_loss = criterion(_classes, _diffs, classes, diffs)
        _, _classes = _classes.max(-1)
        accs = classes == _classes
    return _classes, _diffs, loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()
