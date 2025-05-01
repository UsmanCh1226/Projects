from src.loading_data import load_data
from src.cnn_model_builder import build_model
from src.training_model import train_model
from src.utils import plot_training_history, plot_confusion_matrix

# Constants
base_path = "/Users/usmanchaudhry/PycharmProjects/Projects/MRI Scans/Training"
categories = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 128

# Load data
X_train, X_val, y_train, y_val = load_data(base_path, categories, IMG_SIZE)

# Build model
model = build_model(IMG_SIZE, len(categories))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = train_model(model, X_train, y_train, X_val, y_val)

# Plot results
plot_training_history(history)
plot_confusion_matrix(model, X_val, y_val, categories)