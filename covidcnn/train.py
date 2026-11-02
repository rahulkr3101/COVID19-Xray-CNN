"""Train the CNN on a chest X-ray dataset.

python -m covidcnn.train --data-dir DataSet/Data --epochs 35 --model-out model.h5
"""

from __future__ import annotations

import argparse
import csv

from .data import DataConfig, build_generators
from .model import ModelConfig, build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True,
                    help="Dataset root containing train/ and test/, each with COVID19/ and NORMAL/ subfolders")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=150, help="Images are resized to (size, size)")
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--model-out", default="model.h5")
    p.add_argument("--history-csv", default=None, help="Optional path to write per-epoch metrics as CSV")
    return p.parse_args(argv)


def train(args: argparse.Namespace):
    data_config = DataConfig(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    train_gen, val_gen, test_gen = build_generators(data_config)
    print("Class labels:", train_gen.class_indices)

    model_config = ModelConfig(
        input_shape=(args.image_size, args.image_size, 3),
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    model = build_model(model_config)
    model.summary()

    history = model.fit(train_gen, epochs=args.epochs, validation_data=val_gen)

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")

    if args.history_csv:
        with open(args.history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            keys = list(history.history.keys())
            writer.writerow(["epoch", *keys])
            for i, row in enumerate(zip(*history.history.values())):
                writer.writerow([i, *row])
        print(f"Wrote training history to {args.history_csv}")

    return history, (test_loss, test_acc)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
