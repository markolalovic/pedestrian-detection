#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" model.py: A custom model for CityPersons. """

import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import data

def get_model():
    ''' Returns the model a pretrained model for finetunning on CityPersons. '''

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):
    ''' Converts a PIL image into PyTorch tensor. '''

    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

def save_model(model, path="./models/entire_model.pt"):
    torch.save(model, path)
    print('Model saved to ' + path)

def load_model(path="./models/entire_model.pt"):
    if torch.cuda.is_available():
        return torch.load(path)
    else:
        return torch.load(path, map_location=torch.device('cpu'))

def convert(img, img_raw):
    '''
    Converts the image from dataset back to the raw format:

    * rescales it from  [0,1]  back to  [0,255]  range;
    * flips the channels back to  [height,width,3]  format;
    * converts from tensor to numpy array;
    * converts from numpy array to PIL Image;
    * checks if the image was augmented - flipped horizontally
    '''
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img = np.array(img)
    print('img shape: %d x %d x %d' % img.shape)
    img = Image.fromarray(np.uint8(img)).convert('RGB')

    img_flipped = np.array(img.transpose(Image.FLIP_LEFT_RIGHT))
    img_raw = np.array(img_raw)
    img_was_flipped = np.sum(img_flipped.flatten() == img_raw.flatten()) == img_flipped.shape[0] * img_flipped.shape[1] * img_flipped.shape[2]
    print('Image was flipped: %r' % img_was_flipped)

    return img

## testing on images from Hambrug
if __name__ == "__main__":
    img_path = './datasets/citypersons/hamburg/'
    anno_path = './datasets/citypersons/CityPersons/annotations/'

    # split dataset into train and test
    dataset = data.HamburgDataset(img_path, anno_dict, get_transform(train=True))
    dataset_test = data.HamburgDataset(img_path, anno_dict, get_transform(train=False))

    # permute the indices
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train: 248 - 50 examples
    # test: 50 examples
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    if train:
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        model = get_model()
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # Let's train the model for 10 epochs, evaluating at the end of every epoch.
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

        save(model)
    else:
        model = load_model()

    ## error analysis
    # raw image
    img_raw = Image.open(img_path + imgs[0])
    anno_raw = anno_dict[imgs[0]]

    # same image from the dataset
    idx = indices.index(0)
    img, anno = dataset[idx]
    img = convert_back(img, img_raw)

    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    preds = prediction['boxes'] # predicted bboxes
    preds = preds.cpu().data.numpy() # to numpy array

    scores = prediction['scores'] # scores of predicted bboxes
    scores = scores.cpu().data.numpy()

    # keep only bboxes where score > threshold:
    threshold = .3
    highs = list(np.where(scores > threshold)[0])

    # transform the bboxes from tensor to list and back to [x, y, w, h] format
    bboxes_x0x1y0y1 = []
    for high in highs:
        bboxes_x0x1y0y1.append(list(preds[high]))
    bboxes = []
    for bbox in bboxes_x0x1y0y1:
        bbox = list(bbox)
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]

        bboxes.append([x0, y0, x1 - x0, y1 - y0])

    # draw the predicted bounding boxes
    # TODO: add ground truth bboxes in green
    plt.rcParams['figure.figsize'] = [12, 8]
    fig, ax = plt.subplots()
    ax.imshow(img);

    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    plt.title(img_name)
    plt.show()
