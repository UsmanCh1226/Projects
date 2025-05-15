
import tensorflow as tf
import numpy as np
import cv2


def generate_grad_cam(model, image_tensor, last_conv_layer_name, last_dense_layer_name):
    # Convert the image tensor to a tf.Tensor (in case it's numpy array)
    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)

    # Ensure the model has been built and compiled
    _ = model.predict(image_tensor)

    try:
        # Get the desired layers (Conv and Dense layers)
        conv_layer = model.get_layer(last_conv_layer_name)
        dense_layer = model.get_layer(last_dense_layer_name)
    except ValueError as e:
        raise ValueError(f"Could not find specified layers. {e}")

    # Build the Grad-CAM model with the conv layer and the output layer
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, dense_layer.output]
    )

    # Start recording gradients
    with tf.GradientTape() as tape:
        # Watch the input to the model
        tape.watch(image_tensor)

        # Forward pass to get the convolution output and predictions
        conv_outputs, predictions = grad_model(image_tensor)

        if predictions is None:
            raise ValueError("Predictions returned None.")

        # Get the class index (the class with the highest prediction)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Get the gradients of the loss w.r.t the convolutional outputs
    grads = tape.gradient(loss, conv_outputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    # Check if gradients were computed
    if grads is None:
        raise ValueError("Gradients are None. This usually means the layer is not connected properly to the output.")

    # Average the gradients over all spatial dimensions (height and width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the convolutional outputs from the forward pass
    conv_outputs = conv_outputs[0]

    # Multiply each channel of the convolutional output by the corresponding gradient value
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalize the heatmap and apply the final transformation
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + tf.keras.backend.epsilon())

    # Convert to numpy array and resize it to match the original image size
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (128, 128))

    # Return the Grad-CAM heatmap
    return heatmap
