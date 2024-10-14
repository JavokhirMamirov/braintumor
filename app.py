import cv2
from ultralytics import YOLO
import os
import fnmatch
import matplotlib.pyplot as plt

import matplotlib.patches as patches

model = YOLO("runs/detect/train/weights/best.pt")
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
val_images_path = "datasets/Val/Glioma/images/"


def get_images(folder_path):
    images = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    for root, dirs, files in os.walk(folder_path):
        for extension in image_extensions:
            for filename in fnmatch.filter(files, extension):
                images.append(os.path.join(root, filename))

    return images


def draw_bounding_boxes_from_labels(image):
    """Draws bounding boxes on an image using YOLO label format."""
    img = cv2.imread(image)
    height, width, _ = img.shape
    # Read the label file
    label_path = image.replace("images", "labels").replace(".jpg", ".txt")
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Split each line into its components: class_id, center_x, center_y, width, height
        class_id, center_x, center_y, box_width, box_height = map(float, line.strip().split())

        # Denormalize the coordinates (from 0-1 range to pixel values)
        x1 = int((center_x - box_width / 2) * width)
        y1 = int((center_y - box_height / 2) * height)
        x2 = int((center_x + box_width / 2) * width)
        y2 = int((center_y + box_height / 2) * height)

        color = (176, 37, 184)
        font_color = (255, 255, 255)
        # Draw the bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{classes[int(class_id)]}: {1:.2f}"  # Example: "brain tumor: 0.85"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]

        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 10  # Position above the bounding box
        text_bg_x2 = x1 + text_size[0]
        text_bg_y2 = y1 - 2  # Position at the top of the text
        cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, thickness=cv2.FILLED)

        # Put the label above the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label_text, (x1, y1 - 8), font, font_scale, font_color, font_thickness)
    return img


def display_samples(image_path, yolo_model, classes):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Predict using the YOLO model
    results = yolo_model.predict(image_path, conf=0.1)
    for result in results:

        # Iterate over each detection in the result
        for detection in result.boxes:
            # Extract the bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)  # Ensure coordinates are integers
            conf = detection.conf[0].cpu().numpy()
            cls = detection.cls[0].cpu().numpy()

            # Define a color for the bounding box
            color = (57, 72, 237)  # Purple color
            font_color = (255, 255, 255)


            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Prepare the label text
            label_text = f"{classes[int(cls)]}: {conf:.2f}"  # e.g., "brain tumor: 0.85"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]

            text_bg_x1 = x1
            text_bg_y1 = y1 - text_size[1] - 10  # Position above the bounding box
            text_bg_x2 = x1 + text_size[0]
            text_bg_y2 = y1-2  # Position at the top of the text
            cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, thickness=cv2.FILLED)

            # Put the label above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, label_text, (x1, y1 - 8), font, font_scale, font_color, font_thickness)

    return img


def main():
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    model = YOLO("runs/detect/train/weights/best.pt")
    images = get_images(val_images_path)
    predict_folder = "predict/Glioma"
    for img in images[:10]:
        img1 = draw_bounding_boxes_from_labels(img)
        img2 = display_samples(img, model, classes)
        axs[0].imshow(img1)
        axs[1].imshow(img2)
        axs[0].set_title('Original Image')
        axs[1].set_title('Predicted Image')
        plt.savefig(f"{predict_folder}/{img.replace(val_images_path, "")}")


# result = model.predict(source="test/img_1.png", conf=0.5)
# img = result[0]

#
# img = display_samples("datasets/Val/Glioma/images/gg (21).jpg", model, classes)
# plt.imshow(img)
# plt.axis('off')  # Optionally remove the axis
# plt.show()

if __name__ == '__main__':
    main()
