from ultralytics import YOLO
import torch
model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict('test/img_4.png', save=True, conf=0.01)
# results[0].save('result.jpg')
