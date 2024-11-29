import cv2
import numpy as np

def classify_orientation_with_templates(image_path, templates):
    # Load the target image
    target_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.resize(target_img, (256, 256))  # Normalize size for comparison

    best_match = None
    highest_score = -np.inf

    # Compare with each template
    for orientation, template_path in templates.items():
        # Load the template
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_img = cv2.resize(template_img, (256, 256))  # Normalize size

        # Use normalized cross-correlation for similarity
        res = cv2.matchTemplate(target_img, template_img, cv2.TM_CCOEFF_NORMED)
        score = np.max(res)

        # Track the best match
        if score > highest_score:
            highest_score = score
            best_match = orientation

    return best_match

# Example usage
templates = {
    "Axial": "datasets/Train/Glioma/images/gg (4).jpg",
    "Coronal": "datasets/Train/Glioma/images/gg (134).jpg",
    "Sagittal": "datasets/Train/Glioma/images/gg (76).jpg"
}

image_path = "datasets/Train/Glioma/images/gg (42).jpg"
orientation = classify_orientation_with_templates(image_path, templates)
print(f"The MRI image is classified as: {orientation}")
