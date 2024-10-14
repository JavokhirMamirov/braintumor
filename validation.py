import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import fnmatch
# Load YOLOv8 model
model = YOLO('path/to/your/yolov8/weights.pt')  # Replace with actual path to your model weights

# Load validation dataset (replace with actual paths)
val_images_path = "datasets/Val/Glioma/images/"
val_labels_path = "datasets/Val/Glioma/labels/"

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])  # Box format [x1, y1, x2, y2]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def get_images(folder_path):
    images = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    for root, dirs, files in os.walk(folder_path):
        for extension in image_extensions:
            for filename in fnmatch.filter(files, extension):
                images.append(os.path.join(root, filename))

    return images


val_image_files = get_images(val_images_path)
# Iterate through each image in the validation dataset
for image_file in val_image_files:
    # Load image
    img = cv2.imread(image_file)

    # Real image: Load ground truth labels
    label_path = f"{image_file.replace("images","labels").replace('.jpg', '.txt')}"
    real_boxes = [...]  # Load boxes from the label file (you need to parse them)

    # Predict image: Get predictions from YOLO model
    results = model.predict(image_file)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # YOLOv8 predicted boxes

    # Draw bounding boxes on real and predicted images
    real_img = draw_bounding_boxes(img.copy(), real_boxes, color=(0, 255, 0))  # Green for real
    pred_img = draw_bounding_boxes(img.copy(), pred_boxes, color=(0, 0, 255))  # Red for predictions

    # Plot the real and predicted images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
    plt.title("Real Image with Ground Truth")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    plt.title("Predicted Image")

    plt.show()