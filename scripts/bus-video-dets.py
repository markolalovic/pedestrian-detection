#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# bus-video-dets.py - Generates detections for:
#
#     https://motchallenge.net/vis/MOT16-13/raw/
#
#

import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os

img_path = '/home/marko/data/video/bus-video/frames/'
img_names = list(sorted(os.listdir(img_path)))

img_path_pub = '/home/marko/data/video/bus-video/public-frames/'
img_names_pub = list(sorted(os.listdir(img_path_pub)))

dest_path = '/home/marko/data/video/bus-video/detected-frames/'

with open(r"../data/predictions-bus.pickle", "rb") as file:
    bus_preds = pickle.load(file)

print(len(list(bus_preds.keys())))

threshold = .7
for i in range(len(img_names)):
    img_name = img_names[i]
    print(img_name)

    img = Image.open(img_path + img_name)

    img_name_pub = img_names_pub[i]
    img_pub = Image.open(img_path_pub + img_name_pub)

    pred = bus_preds[img_name]
    highs = list(np.where(pred['scores'] > threshold)[0])
    bboxes_x0x1y0y1 = []

    for high in highs:
        bboxes_x0x1y0y1.append(list(pred['boxes'][high]))

    bboxes = []
    for bbox in bboxes_x0x1y0y1:
        bbox = list(bbox)
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]

        bboxes.append([x0, y0, x1 - x0, y1 - y0])

    plt.rcParams['figure.figsize'] = [12, 8]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gca().set_axis_off()

    plt.subplots_adjust(
        left=0.01,         # position of the left edge of the subplots, as a fraction of the figure width
        right=(1 - 0.01),  # position of the right edge of the subplots, as a fraction of the figure width
        bottom=0.01,
        top=(1 - 0.01),
        wspace=0.1
    )

    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ## enumerate frames
    plt.gcf().text(0.55, 0.65, str(i+1), fontsize=14, color='black')

    text_string = 'Video: MOT Challenge - Pedestrian Detection Challenge, filmed from a bus on a busy intersection at 25fps.'
    plt.gcf().text(0.01, 0.18, text_string, fontsize=14)

    text_string = 'Public detections: taken from  MOT Challenge public detections (https://motchallenge.net/vis/MOT16-13/det/).'
    plt.gcf().text(0.01, 0.14, text_string, fontsize=14)

    text_string = 'Our detections: modified Faster R-CNN trained on CityPersons (threshold=0.7) + MA(3) on Tesla V100 at ~6fps.'
    plt.gcf().text(0.01, 0.1, text_string, fontsize=14)

    ax1.set_axis_off()
    ax2.set_axis_off()

    ax1.imshow(img_pub)
    ax2.imshow(img)

    ax1.set_title('Public detections', fontsize=20)
    ax2.set_title('Our detections', fontsize=20)

    # bbox = [x, y, w, h]
    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=1, edgecolor='red',linestyle ='-', facecolor='none')
        ax2.add_patch(rect)

    fig.savefig(dest_path + img_name, dpi=300, pad_inches = 0,
                facecolor='white', edgecolor='black')
    plt.close()
