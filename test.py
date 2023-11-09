#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @creat_time      : 2023/10/18 10:40
# @author          : lijiantao
# @filename        :  / 
# @description     :

from ultralytics import YOLO
import argparse
from pathlib import Path

my_traindir = ''
my_testdir = ''


# FILE = Path(__file__).resolve()
# # Load a model

# # model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)
#
# # Use the model
# model.train(data="coco128.yaml", epochs=1, batch=4,task='detect',ltest='ltest')  # train the model

#
# print(metrics.box.map)  # map50-95
# print(metrics.box.map50)  # map50
# print(metrics.box.map75)  # map75
#
# print(metrics.box.maps)   # a list contains map50-95 of each category

# results = model("https://ultralytics.com/images/bus.jpg",save=True)  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', type=str, default="runs/yolov/detect/train2/weights/yolov_best.pt", help='模型名称')
    parser.add_argument('--mode', type=str, default='test', help='一个字符串参数，默认值为 "default_value"')
    args = parser.parse_args()
    last_slash_index = args.model.rfind('/')
    # print(last_slash_index)
    if args.model[-2:] == 'pt':
        model = YOLO(args.model)  # build a new model from scratch
        experiment_dir = args.model[:-3] if last_slash_index == -1 else args.model[last_slash_index + 1:-8]
    else:
        model = YOLO(args.model).load('yolov8l.pt')
        experiment_dir = args.model[:-5]
    if args.mode == 'test':
        metrics = model.val(data="coco128.yaml", split='val', experiment_dir=experiment_dir,
                            save_txt=True,save_json=True)  # evaluate model performance on the validation set
        # 定义每行要输出的元素数量
        elements_per_line = 3

        for i, item in enumerate(metrics.box.maps):
            # 输出当前元素
            print(item, end=" ")

            # 检查是否需要换行
            if (i + 1) % elements_per_line == 0:
                print()  # 添加换行符

        # 如果列表长度不是3的倍数，最后可能需要额外的换行
        if len(metrics.box.maps) % elements_per_line != 0:
            print()
    elif args.mode == 'train':

        model.train(data="coco128.yaml", epochs=1, batch=4, task='detect', experiment_dir=experiment_dir)


if __name__ == '__main__':
    main()
