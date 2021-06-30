#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" eval.py: Evaluation using COCO tools.

TODO: Write a function that converts FasterRCNN results to detections.json file.

"""

import sys
sys.path.insert(0, '../src/coco_evaluation_tools/')
from MR_coco import COCO
from MR_eval_multisetup import COCOeval
from contextlib import redirect_stdout

## set annotations type and paths to files
annType = 'bbox'
annFile = '../data/data0.json'
resFile = '../data/det0.json'

## testing the evaluation
if __name__ == "__main__":
    res_file = open("../data/results.txt", "w")
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        cocoEval.summarize(id_setup, res_file)

    res_file.close()
