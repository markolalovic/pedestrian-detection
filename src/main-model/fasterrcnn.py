#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" rcnn.py: main faster R-CNN model using torchvision implementation """

import numpy as np
from PIL import Image
import torchvision.models as models
import torch
import torch.nn as nn

from fasterutils import get_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, imgs, anno_dict, transforms=None):
        self.img_paths = img_paths
        self.transforms = transforms
        self.imgs = imgs
        self.anno = anno_dict

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        img = Image.open(img_path).convert('RGB')

        ## prepare bboxes coordinates
        ## transform from [x, y, w, h] to [x0, y0, x1, y1]
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
