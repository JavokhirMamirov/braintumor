from ultralytics import YOLO
import torch
model = YOLO("C:/Projects/brain_tumor/yolo_train/runs/detect/train3/weights/best.pt")

results = model.predict('test/img_1.png', conf=0.01)
results[0].save('result.jpg')
