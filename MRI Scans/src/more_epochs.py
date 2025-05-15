import subprocess
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from cnn_model_builder import build_model
from loading_data import loading_data
from logs import get_logger
import os

BASE_DIR = os.path.dirname(__file__)


def rel_path(*parts):
    return os.path.join(BASE_DIR, *parts)

logger = get_logger(__name__)

def main():
    try:
        logger.info("Starting training pipeline...")

        # Load datasets
        train_data, val_data = loading_data()
        logger.info("Datasets loaded.")

        # Log sample shapes
        for x, y in train_data.take(1):
            logger.info(f"Train input shape: {x.shape}")
        for x, y in val_data.take(1):
            logger.info(f"Val input shape: {x.shape}")

        # Build model (ensure IMG_SIZE and num_classes match your data)
        IMG_SIZE = 128
        NUM_CLASSES = 3
        model = build_model(IMG_SIZE, NUM_CLASSES)
        logger.info("Model built.")
        model.summary()

        # Compute class weights
        y_train = np.concatenate([y.numpy() for _, y in train_data], axis=0).argmax(axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))

        # Callbacks
        checkpoint_path = rel_path('..', 'checkpoints', 'best_weights.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = ModelCheckpoint(checkpoint_path, ...)


        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )

        # Training
        history = model.fit(
            train_data,
            epochs=100,
            validation_data=val_data,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            class_weight=class_weights_dict,
            verbose=1
        )

        weights_path = rel_path('..', 'models', 'brain_tumor_model.weights.h5')
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        model.save_weights(weights_path)
        logger.info("Model training complete and weights saved.")

        history_path = rel_path('..', 'artifacts', 'training_history.pkl')
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        logger.info("Training history saved.")

    except Exception as e:
        logger.error(f"Error during training: {e}")

if __name__ == '__main__':
    proc = subprocess.Popen(["caffeinate"])
    try:
        main()
    finally:
        proc.terminate()
        logger.info("Caffeinate process terminated.")
