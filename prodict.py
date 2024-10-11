import fnmatch

from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import cv2
import os
import imutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches

classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


# Function to display detection results with highlighted parts
def display_samples(images, yolo_model):
    for i in range(len(images)):
        img = images[i]
        result = yolo_model.predict(img)[0]  # Assuming batch size of 1, take the first result

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        ax = plt.gca()

        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            conf = detection.conf[0].cpu().numpy()
            cls = detection.cls[0].cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f"{classes[int(cls)]} {conf:.2f}", color='white', fontsize=12, backgroundcolor='red')

        plt.title(f'YOLOv8 Detection')
        plt.show()


def get_images():
    images = []
    folder_path = 'test'
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    for root, dirs, files in os.walk(folder_path):
        for extension in image_extensions:
            for filename in fnmatch.filter(files, extension):
                images.append(os.path.join(root, filename))

    return images


if __name__ == '__main__':
    # Load a model
    model = YOLO("C:/Projects/brain_tumor/yolo_train/runs/detect/train3/weights/best.pt")  # load a custom model
    images = get_images()
    model.predict(source=images, save=True, conf=0.01)

