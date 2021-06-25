import json
import numpy as np
import torch

def results_as_json(model, indices, dataset, device, num_im, threshold = 0.3):
    infos = []
    for i in range(0, num_im): #i don't know where to find the number of images so I just added a variable
        idx = indices.index(i)
        img, anno = dataset[idx]
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])[i]

        preds = prediction['boxes']  # predicted bboxes
        preds = preds.cpu().data.numpy()  # to numpy array
        scores = prediction['scores']  # scores of predicted bboxes
        scores = scores.cpu().data.numpy()

        highs = list(np.where(scores > threshold)[0])

        for high in highs:
            bb = list(preds[high])
            sc = scores[high]

            info = {"image_id": idx, #this idx should be the same as the one in the data
                    "category_id": 1,
                    "bbox": bb,
                    "score": sc}
            infos.append(info)

    f = open("results.json","w")
    f. write(json.dumps(infos))