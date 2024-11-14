import cv2
import os
import shutil
from skimage.metrics import structural_similarity as ssim

similarity_threshold = 0.65
smiler_images = ['3.jpg', '1.jpg', 'image (14).jpg', 'image(15).jpg', 'image(42).jpg', 'image(52).jpg',
                 'Tr-no_0215.jpg', 'Tr-no_0298.jpg', 'Tr-no_0605.jpg']
folder_images = 'datasets/Train/Pituitary/images'
folder_labels = 'datasets/Train/Pituitary/labels'
folder_image_save = 'datasets3/Axial/Train/Pituitary/images'
folder_label_save = 'datasets3/Axial/Train/Pituitary/labels'
# first_image = os.path.join(folder_images, 'gg (2).jpg')
for filename in os.listdir(folder_images):
    if filename.endswith(".jpg"):
        img_path = os.path.join(folder_images, filename)
        label_path = os.path.join(folder_labels, filename.replace(".jpg", ".txt"))
        for img in smiler_images:
            first_image = os.path.join(folder_images, img)
            imageA = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
            imageB = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if imageA.shape != imageB.shape:
                pass
            else:
                score, diff = ssim(imageA, imageB, full=True)
                diff = (diff * 255).astype("uint8")
                if score >= similarity_threshold:
                    shutil.copy(img_path, os.path.join(folder_image_save, filename))
                    shutil.copy(label_path, os.path.join(folder_label_save, filename.replace(".jpg", ".txt")))
                    print(f"(SSIM: {score:.2f})")
                    break
