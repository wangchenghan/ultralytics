'''
Author: wangchenghan chenghan.wang@ihandysoft.com
Date: 2024-09-28 20:50:54
LastEditors: wangchenghan chenghan.wang@ihandysoft.com
LastEditTime: 2024-09-28 21:24:44
FilePath: /fish_and_vegetable/scripts/train/train_yolov8_obb.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# conda activate yolo8
# pip install ultralytics

from ultralytics import YOLO

# Load a model
model = YOLO("../models/yolov8n-obb.pt")

# Train the model
train_results = model.train(
    data="ultralytics/cfg/datasets/fish_obb.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("../datasets/fish_obb/images/test/00010000998000000_frame64105.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model