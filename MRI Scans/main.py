import os
import numpy as np
import json
from src.loading_data import loading_data
from src.cnn_model_builder import build_model
from src.training_model import train_model
from src.utils import plot_training_history, plot_confusion_matrix
from src.logs import get_logger
from sklearn.metrics import classification_report

IMG_SIZE = 128
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
base_path = os.path.join(os.path.dirname(__file__), '..', 'Training')

logger = get_logger(__name__)

def main():
    logger.info("Starting the pipeline...")

    # Load data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = loading_data(base_path, categories, img_size=IMG_SIZE)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Build model
    try:
        model = build_model(IMG_SIZE, len(categories))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model built and compiled.")
    except Exception as e:
        logger.error(f"Error building model: {e}")
        return

    # Train model
    try:
        history = train_model(model, X_train, y_train, X_val, y_val)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return

    # Save the best model (already handled by checkpoint)
    os.makedirs('models', exist_ok=True)
    logger.info("Model saved successfully.")

    # Plot training history
    plot_training_history(history)

    # Evaluate on validation set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n✅ Test Accuracy: {test_accuracy:.2f} | Test Loss: {test_loss:.2f}")

    # Plot confusion matrix
    plot_confusion_matrix(model, X_val, y_val, categories)

    # Log class-wise metrics
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=categories)
    print(report)
    logger.info("Pipeline completed successfully.")


if __name__ == '__main__':
    main()
