"""Commands for training livestock detection models."""

import argparse
from pathlib import Path


def train_model(args):
    from ultralytics import YOLO

    print(f"[TRAIN] Loading model: {args.model}")
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        name=args.run_name,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[TRAIN] Finished training. Best weights: {best_weights}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train livestock detection models with Ultralytics YOLO"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a YOLO model")
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--data", default="merged_ds/dataset.yaml")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="cpu")
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-name", default="train")
    train.set_defaults(func=train_model)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
