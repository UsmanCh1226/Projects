import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
from streamlit import graphviz_chart
from tensorflow.keras.models import load_model
import os
from grad_cam import generate_grad_cam
import io
import sys
from tensorflow.keras.utils import plot_model

def rel_path(*path_parts):
    return os.path.join(os.path.dirname(__file__), *path_parts)


def load_trained_model():
    model_path = rel_path('..', 'models', 'best_model.keras')
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

IMG_SIZE = 128
CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor Classifier")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=['jpg', 'png', 'jpeg'])

model = load_trained_model()

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_tensor = np.expand_dims(resized_image, axis=0) / 255.0  # Normalize image

    # Convert to tf.Tensor explicitly
    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)

    # Predict
    preds = model.predict(image_tensor)
    pred_label = CATEGORIES[np.argmax(preds)]

    st.image(image, caption=f"🧠 Prediction: {pred_label}", use_container_width=True)

    try:
   
        last_conv_layer_name = 'conv2d_2' 
        last_dense_layer_name = 'dense_1' 

        # Generate Grad-CAM
        cam = generate_grad_cam(model, image_tensor, last_conv_layer_name, last_dense_layer_name)

        # Apply heatmap to the original image
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(resized_image, 0.6, heatmap, 0.4, 0)
        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not generate Grad-CAM: {e}")

def get_model_summary(model):
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    return stream.getvalue()

with st.expander("🧠 Show Model Summary"):
    summary_str = get_model_summary(model)
    st.text(summary_str)
