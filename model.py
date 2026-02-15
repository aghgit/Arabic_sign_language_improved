import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Bloc 1
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            # Bloc 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            # Bloc 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            # Classification head (allégé)
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model
