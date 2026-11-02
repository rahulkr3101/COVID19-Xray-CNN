"""The CNN architecture: 2x (Conv -> Pool -> Dropout) -> Dense -> sigmoid."""

from __future__ import annotations

from dataclasses import dataclass

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


@dataclass
class ModelConfig:
    input_shape: tuple[int, int, int] = (150, 150, 3)
    dropout: float = 0.5
    dense_units: int = 256
    learning_rate: float = 0.001


def build_model(config: ModelConfig) -> Sequential:
    model = Sequential([
        Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=config.input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(config.dropout),

        Conv2D(64, (5, 5), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(config.dropout),

        Flatten(),
        Dense(config.dense_units, activation="relu"),
        Dropout(config.dropout),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
