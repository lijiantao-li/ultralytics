#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @creat_time      : 2023/10/18 10:40
# @author          : lijiantao
# @filename        :  / 
# @description     :

from ultralytics import YOLO
import argparse
import os
from pathlib import Path

my_traindir = ''
my_testdir = ''


def find_latest_folder(directory):
    # 获取目录中所有文件夹的列表
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # 获取每个文件夹的创建时间
    folder_creation_times = [(folder, os.path.getctime(os.path.join(directory, folder))) for folder in folders]

    # 按创建时间排序
    folder_creation_times.sort(key=lambda x: x[1], reverse=True)

    # 返回最新创建的文件夹的路径
    if folder_creation_times:
        latest_folder = folder_creation_times[0][0]
        return os.path.join(directory, latest_folder)
    else:
        return None


#
# print(metrics.box.map)  # map50-95
# print(metrics.box.map50)  # map50
# print(metrics.box.map75)  # map75
#

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', type=str, default="runs/yolov8l/train/weights/yolov8l_best.pt", help='模型名称')
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
        directory_to_search = f"./runs/{experiment_dir}/val"
        latest_folder = find_latest_folder(directory_to_search)

        if latest_folder:
            print(f"The latest folder in {directory_to_search} is: {latest_folder}")
        else:
            print(f"No folders found in {directory_to_search}")
        # metrics = model.val(data="myVisDrone.yaml", split='val', experiment_dir=experiment_dir,
        #                     save_txt=True, save_json=True)  # evaluate model performance on the validation set
        #
        # print(metrics.box.maps)
        # print(metrics)
    elif args.mode == 'train':

        model.train(data="myVisDrone.yaml", epochs=1, batch=4, task='detect', experiment_dir=experiment_dir)


# 用法示例


if __name__ == '__main__':
    main()
