import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(base_path, categories, img_size=128):
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
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"⚠️ Error with image {img_path}: {e}")

    X = np.array(data, dtype='float32') / 255.0  # Normalize
    y = to_categorical(labels, num_classes=len(categories))  # One-hot encoding

    return train_test_split(X, y, test_size=0.2, random_state=42)


