import argparse
import csv
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from demo_yolo.yolo_pipeline import predict_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = Path(__file__).resolve().parent / "static" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ALLOWED_IMAGES = {".jpg", ".jpeg", ".png"}

MODELS = {
    "yolo8n.pt": "YOLO8 Nano",
    "yolo8s.pt": "YOLO8 Small",
    "yolo8m.pt": "YOLO8 Medium",
    "yolo8l.pt": "YOLO8 Large",
    "yolo10n.pt": "YOLO10 Nano",
    "yolo10s.pt": "YOLO10 Small",
    "yolo10m.pt": "YOLO10 Medium",
    "yolo11n.pt": "YOLO11 Nano",
    "yolo11s.pt": "YOLO11 Small",
    "yolo11m.pt": "YOLO11 Medium",
    "yolo11l.pt": "YOLO11 Large",
    "yolo11x.pt": "YOLO11 XLarge",
}


def clear_folder(folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def count_summary(csv_path):
    if not csv_path or not csv_path.exists():
        return {}

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    totals = []
    for row in rows:
        total = 0
        for name, value in row.items():
            if name != "image":
                total += int(value or 0)
        totals.append(total)

    if not totals:
        return {}
    return {
        "num_images": len(totals),
        "total_detections": sum(totals),
        "mean_detections_per_image": round(sum(totals) / len(totals), 2),
        "max_detections": max(totals),
        "min_detections": min(totals),
    }


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html", models=MODELS)

    @app.post("/upload")
    def upload_images():
        files = request.files.getlist("files[]")
        uploaded = []
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for file in files:
            filename = secure_filename(file.filename or "")
            if not filename or Path(filename).suffix.lower() not in ALLOWED_IMAGES:
                continue
            file.save(UPLOAD_DIR / filename)
            uploaded.append(filename)

        if not uploaded:
            return jsonify({"error": "No valid images were uploaded"}), 400
        return jsonify({"uploaded": uploaded})

    @app.post("/predict")
    def predict_images():
        data = request.get_json(silent=True) or {}
        model_name = data.get("model", "yolo11s.pt")
        if model_name not in MODELS:
            return jsonify({"error": "Unknown model selected"}), 400

        requested = data.get("files") or []
        uploaded = [
            UPLOAD_DIR / secure_filename(name)
            for name in requested
            if (UPLOAD_DIR / secure_filename(name)).is_file()
        ]
        if not uploaded:
            return jsonify({"error": "No uploaded images to predict"}), 400

        checkpoint = CHECKPOINT_DIR / model_name
        if not checkpoint.is_file():
            return jsonify({"error": f"Checkpoint not found: {model_name}"}), 404

        try:
            conf = float(data.get("conf", 0.25))
            iou = float(data.get("iou", 0.7))
            if not 0 <= conf <= 1 or not 0 <= iou <= 1:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "Confidence and IoU must be between 0 and 1"}), 400

        clear_folder(RESULTS_DIR)
        args = argparse.Namespace(
            source=[str(path) for path in uploaded],
            weights=str(checkpoint),
            run_name="train",
            output_name="gui_predict",
            conf=conf,
            iou=iou,
            imgsz=640,
            device=data.get("device", "cpu"),
            save=True,
            save_txt=False,
        )

        try:
            results, csv_path = predict_model(args)
        except Exception as error:
            return jsonify({"error": f"Prediction failed: {error}"}), 500

        result_urls = []
        if results:
            save_dir = Path(results[0].save_dir)
            for source in sorted(save_dir.iterdir()):
                if source.suffix.lower() in ALLOWED_IMAGES:
                    target = RESULTS_DIR / source.name
                    shutil.copy2(source, target)
                    result_urls.append(f"/static/results/{target.name}")

        return jsonify(
            {
                "success": True,
                "images": result_urls,
                "metrics": count_summary(csv_path),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
