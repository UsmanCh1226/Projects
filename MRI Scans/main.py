import kaggle
import numpy as np
import pandas as pd
import random
import os
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


kaggle.api.authenticate()

kaggle.api.dataset_download_files("masoudnickparvar/brain-tumor-mri-dataset",path='.',unzip=True)

kaggle.api.dataset_metadata("masoudnickparvar/brain-tumor-mri-dataset",path='.')

#print(kaggle.api.dataset_list_files("masoudnickparvar/brain-tumor-mri-dataset").files)


base_path = "/Users/usmanchaudhry/PycharmProjects/Projects/MRI Scans/Training"
print(f"Base path: {base_path}")
categories = ["glioma", "meningioma", "no tumor", "pituitary"]

IMG_SIZE = 128
data = []
labels = []

for idx, category in enumerate(categories):
    category_path = os.path.join(base_path, category)

    if not os.path.exists(category_path):
        print(f"Folder not found: {category_path}")
    else:
        print(f"Reading images from: {category_path}")

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(idx)
        except Exception as e:
            print(f"Error with image {img_path}: {e}")


X = np.array(data) / 255.0
y = to_categorical(labels, num_classes = 4)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Samples: {len(X_train)}")
print(f"Validation Samples: {len(X_val)}")


