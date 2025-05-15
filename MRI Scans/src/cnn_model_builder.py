from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras import regularizers


def build_model(IMG_SIZE, num_classes):
    """
    Build and return a Sequential CNN model.

    Parameters:
    IMG_SIZE (int): The size of the input image dimensions (IMG_SIZE x IMG_SIZE)
    num_classes (int): The number of output classes.

    Returns:
    model (Sequential): A compiled Keras Sequential model.
    """
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model
