# YOLO-NAS vs YOLOV8 for Real-time Object Detection - Pros and Cons
# https://www.youtube.com/watch?v=_ON9oiT_G0w

# WORKING

from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")              # s,m,l or x

results = model.predict(source="0", show=True)
# help(model)