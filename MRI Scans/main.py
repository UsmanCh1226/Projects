import os
from difflib import restore

import cv2
import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.legacy.backend import dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

#Preprocessing Data

#Normalization (change values of columns to 0s and 1s)
X = np.array(data, dtype='float32') / 255.0

#Encoding Labels
y = to_categorical(labels, num_classes=len(categories))

#Train/Test Split

X_train, X_value, y_train, y_value = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Samples: {len(X_train)}")
print(f"Validation Samples: {len(X_value)}")

#Building CNN Model

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))

model.summary()

#Compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#Training the model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_value, y_value),
                    epochs=20,
                    batch_size=32,
                    callbacks=[early_stop])

def plot_sample_images(X, y, categories, samples_per_category=5):
    plt.figure(figsize=(15,10))

    for idx, category in enumerate(categories):
        category_indices = np.where(np.argmax(y, axis=1) == idx[0])

        for i in range(samples_per_category):
            img_index = category_indices[i]
            plt.subplot(len(categories), samples_per_category, idx * samples_per_category + i + 1)
            plt.imshow(X[img_index])
            plt.title(category)
            plt.axis('off')


    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14,6))

    #Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title("Training and Validation Accuracy")

    #Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title("Training and Validation Accuracy")

    plt.show()
    f








