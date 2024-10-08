'''
Author: wangchenghan chenghan.wang@ihandysoft.com
Date: 2024-09-28 20:50:54
LastEditors: wangchenghan chenghan.wang@ihandysoft.com
LastEditTime: 2024-09-29 01:00:10
FilePath: /fish_and_vegetable/scripts/train/train_yolov8_obb.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# conda activate yolo8
# pip install ultralytics

from ultralytics import YOLO

# Load a model
model = YOLO("runs/obb/train/weights/best.onnx")

# Perform object detection on an image
results = model("../datasets/fish_obb/images/test/00010000998000000_frame64105.jpg")
print(dir(results[0]))
results[0].show(font_size=20)
results[0].save('result.jpg')