#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @creat_time      : 2023/10/18 10:40
# @author          : lijiantao
# @filename        :  / 
# @description     :

from ultralytics import YOLO

# Load a model
model = YOLO("myyolov8l_atten.yaml") # build a new model from scratch
# model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=1, batch=4,task='detect')  # train the model
# metrics = model.val(data="coco128.yaml",)  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg",save=True)  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
