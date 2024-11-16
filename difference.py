import cv2
from ultralytics import YOLO
import os
import fnmatch
import matplotlib.pyplot as plt


def main():
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    model_1 = YOLO("runs/detect/train/weights/best.pt")
    model_2 = YOLO("runs/detect/train4/weights/best.pt")
    image_path = "test/img_2.png"
    result_1 = model_1.predict(image_path, conf=0.3)
    result_2 = model_2.predict(image_path, conf=0.3)
    for img in images[:10]:
        img1 = draw_bounding_boxes_from_labels(img)
        img2 = display_samples(img, model, classes)
        axs[0].imshow(img1)
        axs[1].imshow(img2)
        axs[0].set_title('Original Image')
        axs[1].set_title('Predicted Image')
        plt.savefig(f"{predict_folder}/{img.replace(val_images_path, "")}")