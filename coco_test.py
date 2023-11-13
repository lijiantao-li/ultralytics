from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

'''
data:2022.4.6
check:Y
function:验证标注的数据集和预测的数据集
如果格式不对，则会报错
如果无误，会返回检测结果
'''

# accumulate predictions from all images
# 载入coco2017验证集标注文件
# coco_true = COCO(annotation_file='coco/train.json')
coco_true = COCO(annotation_file=r'/home/li/桌面/Yolo-to-COCO-format-converter/test/VisDrone2019-DET_val_coco.json')  # 标准数据集（真值）
# coco_pre = coco_true.loadRes('predictions.json')
coco_pre = coco_true.loadRes('/home/li/桌面/ultralytics/runs/yolov8l/val4/predictions.json')  # 预测数据集（预测值）
# 载入网络在coco2017验证集上预测的结果
coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")  # 计算bbox值
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
