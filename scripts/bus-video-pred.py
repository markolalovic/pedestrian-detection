#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# bus-video-pred.py - Generates predictions for:
#
#     https://motchallenge.net/vis/MOT16-13/raw/
#
#

import torch
import torch.utils.data
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(),])
import time
from PIL import Image

model = torch.load(
    f = '/home/marko/data/models/final-model-fit.pt',
    map_location = torch.device('cpu'));

tstart = time.time()
bus_preds = {}
for img_name in img_names:
    img = Image.open(img_path + img_name)

    model.eval()
    img_tr = torch.unsqueeze(transform(img), 0)
    with torch.no_grad():
        pred = model(img_tr)[0]

    boxes, labels, scores = pred['boxes'], pred['labels'], pred['scores']
    boxes, labels, scores = [arr.cpu().detach().numpy() for arr in [boxes, labels, scores]]
    boxes = np.round(boxes)
    boxes = boxes.astype('int64')
    scores = scores.astype('float64')
    pred = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }
    bus_preds[img_name] = pred

tend = time.time()
print('\nExported %d predictions' % (len(bus_preds)))
print('\nTime elapsed = %.2f min' % ((tend - tstart)/60))

with open(r"../data/predictions-bus.pickle", "wb") as output_file:
    pickle.dump(bus_preds, output_file)
