import subprocess
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from cnn_model_builder import build_model
from loading_data import loading_data
from logs import get_logger

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
        checkpoint = ModelCheckpoint(
            'checkpoints/best_weights.h5',
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min'
        )
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

        # Save final weights and training history
        model.save_weights('brain_tumor_model.weights.h5')
        logger.info("Model training complete and weights saved.")

        with open('training_history.pkl', 'wb') as f:
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
