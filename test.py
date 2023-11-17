#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @creat_time      : 2023/10/18 10:40
# @author          : lijiantao
# @filename        :  / 
# @description     :

from ultralytics import YOLO
import argparse
import os
import sys
from pathlib import Path
import shutil

my_traindir = ''


def copy_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"文件已成功复制从 {source_path} 到 {destination_path}")
    except Exception as e:
        print(f"复制文件时出错: {e}")


# 用法示例


def find_latest_folder_with_keyword(directory, keyword):
    # 获取目录中所有文件夹的列表
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # 获取每个文件夹的创建时间和名称
    folder_creation_times = [(folder, os.path.getctime(os.path.join(directory,
                                                                    folder))) for folder in folders]

    # 按创建时间排序
    folder_creation_times.sort(key=lambda x: x[1], reverse=True)

    # 遍历排序后的文件夹列表，找到包含指定关键字的文件夹
    for folder, _ in folder_creation_times:
        if keyword in folder:
            return os.path.join(directory, folder)

    # 如果没有找到匹配的文件夹，则返回None
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

    directory_to_search = f"./runs/{experiment_dir}"
    if args.mode == 'test':

        metrics = model.val(data="myVisDrone.yaml", split='val', experiment_dir=experiment_dir,
                            save_txt=True, save_json=True)  # evaluate model performance on the validation set
        print(metrics.box.maps)
        print(metrics)
        my_testdir = find_latest_folder_with_keyword(directory_to_search, "val")

        copy_file('test.log', fr"{my_testdir}/test.log")
        print(f"The latest folder in {directory_to_search} is: {my_testdir}")
        with open(f'{my_testdir}/myresults.txt', 'w') as f:
            f.write(f"权重文件：{args.model}\n ")
            f.write('总map50:' + str(metrics.box.map50) + '\n')
            f.write('总map75:' + str(metrics.box.map75) + '\n')
            f.write('总p：'+str(metrics.box.mp)+'\n')
            f.write('总r：'+str(metrics.box.mr)+'\n')
            f.write('总map50-95:' + str(metrics.box.map) + '\n')
            f.write('各类map50:' + str(metrics.box.ap50) + '\n')
            f.write('各类map50-95:' + str(metrics.box.ap) + '\n')
            f.write('各类p:'+str(metrics.box.p)+'\n')

            f.write('speed\n')
            f.write('pre: ' + str(metrics.speed['preprocess']) + 'ms\n')
            f.write('inference: ' + str(metrics.speed['inference']) + 'ms\n')
            f.write('post: ' + str(metrics.speed['postprocess']) + 'ms\n')
            f.write('loss: ' + str(metrics.speed['loss']) + 'ms\n')
            f.write('fps: ' + str(1000/(metrics.speed['preprocess']+metrics.speed['inference']+metrics.speed['postprocess']+metrics.speed['loss'])) + '\n')
            f.write('confusion_matrix:'+str(metrics.confusion_matrix) + '\n')
            f.write(str(metrics))

    elif args.mode == 'train':

        model.train(data="myVisDrone.yaml", epochs=1, batch=4, task='detect', experiment_dir=experiment_dir)


# 用法示例


if __name__ == '__main__':
    main()
