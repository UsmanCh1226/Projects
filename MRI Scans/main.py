import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



from src.loading_data import extract_images

#Image Processing
base_path = "/Users/usmanchaudhry/PycharmProjects/Projects/MRI Scans/Training"
print(f"Base path: {base_path}")
categories = ["glioma", "meningioma", "no tumor", "pituitary"]

IMG_SIZE = 128
data = []
labels = []

for idx, category in enumerate(categories):
    category_path = os.path.join(base_path, category)

    if not os.path.isdir(category_path):
        print(f"⚠️ Skipping: {category_path} is not a valid directory.")
        continue

    print(f"✅ Reading images from: {category_path}")

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(idx)
        except Exception as e:
            print(f"⚠️ Error with image {img_path}: {e}")

#CNN Model
