#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" eval.py: Evaluation using COCO tools.

TODO: Write a function that converts FasterRCNN results to detections.json file.

"""

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from contextlib import redirect_stdout

## set annotations type and paths to files
annType = 'bbox'
annFile = '../data/truths.json'
resFile = '../data/detections.json'

## testing the evaluation
if __name__ == "__main__":
    cocoGt=COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()

    with open('../data/results.txt', 'w') as f:
        with redirect_stdout(f):
            cocoEval.summarize()
