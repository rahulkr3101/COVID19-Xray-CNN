"""Train/validation/test data generators for the chest X-ray dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass

from tensorflow.keras.preprocessing.image import ImageDataGenerator


@dataclass
class DataConfig:
    data_dir: str
    image_size: tuple[int, int] = (150, 150)
    batch_size: int = 32
    validation_split: float = 0.2


def build_generators(config: DataConfig):
    """Return (train, validation, test) generators.

    Expects `data_dir` laid out as:
        train/COVID19/*, train/NORMAL/*, test/COVID19/*, test/NORMAL/*
    which is how the original Kaggle-sourced dataset is structured.
    """
    train_dir = os.path.join(config.data_dir, "train")
    test_dir = os.path.join(config.data_dir, "test")

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=config.validation_split,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_aug.flow_from_directory(
        train_dir,
        target_size=config.image_size,
        subset="training",
        batch_size=config.batch_size,
        class_mode="binary",
    )
    validation_generator = train_aug.flow_from_directory(
        train_dir,
        target_size=config.image_size,
        subset="validation",
        batch_size=config.batch_size,
        class_mode="binary",
    )
    test_generator = test_aug.flow_from_directory(
        test_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode="binary",
    )

    return train_generator, validation_generator, test_generator
