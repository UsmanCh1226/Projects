from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from tensorflow.keras.models import load_model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the given model with the provided training and validation data.

    Parameters:
    - model (tensorflow.keras.Model): The model to be trained.
    - X_train (array-like): Training features.
    - y_train (array-like): Training labels.
    - X_val (array-like): Validation features.
    - y_val (array-like): Validation labels.

    Returns:
    - history (History): The history object containing training information.
    """
    os.makedirs("models", exist_ok=True)

    # Define callbacks
    checkpoint = ModelCheckpoint(
        filepath="MRI Scans/models/best_model.keras",  # Save the best model based on val_loss
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[early_stop, reduce_lr, checkpoint],  # Use the callbacks to save the best model
        verbose=1
    )


    return history
