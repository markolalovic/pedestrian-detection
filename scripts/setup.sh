#!/bin/bash
# basic setup script

## dependencies
pip install --upgrade pip
pip install torchvision
pip install cython
pip install pycocotools

## data
mkdir datasets
cd datasets

# for now citypersons dataset only
mkdir citypersons
cd citypersons

# for now hamburg images only
mkdir hamburg
cd hamburg
# download the images from our data server
scp user1@159.89.9.230:/home/user1/cityscapes-dataset/leftImg8bit/train/hamburg/* .
cd ..

# get annotations
git clone https://github.com/cvgroup-njust/CityPersons.git





