# YOLO-NAS vs YOLOV8 for Real-time Object Detection - Pros and Cons
# https://www.youtube.com/watch?v=_ON9oiT_G0w

# NOT WORKING YET

import torch
from super_gradients.training import models

yolo_nas_l = models.get("yolo_nas_s", pretrained_weights="coco")

model = yolo_nas_l("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

model.predict_webcam()