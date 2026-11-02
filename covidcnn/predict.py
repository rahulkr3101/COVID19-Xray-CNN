"""Run a trained model on one or more chest X-ray images.

python -m covidcnn.predict --model model.h5 image1.jpg image2.jpg
"""

from __future__ import annotations

import argparse

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


def predict_image(model, img_path: str, image_size: tuple[int, int] = (150, 150)) -> float:
    """Return the raw sigmoid output for one image (< 0.5 == COVID-19 positive)."""
    img = keras_image.load_img(img_path, target_size=image_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return float(model.predict(arr, verbose=0)[0, 0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to a saved .h5 model")
    p.add_argument("--image-size", type=int, default=150)
    p.add_argument("images", nargs="+", help="Paths to chest X-ray images")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model = load_model(args.model)

    for path in args.images:
        score = predict_image(model, path, (args.image_size, args.image_size))
        label = "COVID-19 positive" if score < 0.5 else "normal"
        print(f"{path}: {label}  (raw score={score:.4f})")


if __name__ == "__main__":
    main()
