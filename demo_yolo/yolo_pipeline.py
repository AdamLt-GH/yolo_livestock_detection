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


def find_weights(weights, run_name):
    if weights:
        return Path(weights)
    return Path("runs") / "detect" / run_name / "weights" / "best.pt"


def validate_model(args):
    from ultralytics import YOLO

    weights = find_weights(args.weights, args.run_name)
    print(f"[VAL] Loading weights: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
        verbose=True,
    )

    values = metrics.results_dict
    summary = {
        "Precision": values.get("metrics/precision(B)"),
        "Recall": values.get("metrics/recall(B)"),
        "mAP@0.50": values.get("metrics/mAP50(B)"),
        "mAP@0.50:0.95": values.get("metrics/mAP50-95(B)"),
    }

    print("[VAL] Results:")
    for name, value in summary.items():
        shown = f"{value:.3f}" if value is not None else "not available"
        print(f"  {name}: {shown}")
    print(f"  Saved to: {metrics.save_dir}")


def predict_model(args):
    from ultralytics import YOLO

    weights = find_weights(args.weights, args.run_name)
    print(f"[PREDICT] Loading weights: {weights}")

    model = YOLO(str(weights))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        stream=False,
        name=args.output_name,
        verbose=True,
    )

    if not results:
        print("[PREDICT] No results returned")
        return

    print(f"[PREDICT] Finished {len(results)} prediction(s)")
    print(f"[PREDICT] Results saved to: {results[0].save_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train, validate and run livestock detection models"
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

    validate = commands.add_parser("val", help="Validate a trained YOLO model")
    validate.add_argument("--weights")
    validate.add_argument("--run-name", default="train")
    validate.add_argument("--data", default="merged_ds/dataset.yaml")
    validate.add_argument("--split", choices=["val", "test"], default="test")
    validate.add_argument("--device", default="cpu")
    validate.set_defaults(func=validate_model)

    predict = commands.add_parser("predict", help="Run predictions on images or videos")
    predict.add_argument("--source", required=True)
    predict.add_argument("--weights")
    predict.add_argument("--run-name", default="train")
    predict.add_argument("--output-name", default="predict")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.7)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    predict.add_argument("--save-txt", action="store_true")
    predict.set_defaults(func=predict_model)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
