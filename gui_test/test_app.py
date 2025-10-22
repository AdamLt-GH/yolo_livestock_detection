import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gui import app as app_module


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.uploads = self.root / "images"
        self.results = self.root / "static" / "results"
        self.checkpoints = self.root / "checkpoints"
        self.run_dir = self.root / "runs" / "detect" / "gui_predict"

        self.patches = [
            patch.object(app_module, "UPLOAD_DIR", self.uploads),
            patch.object(app_module, "RESULTS_DIR", self.results),
            patch.object(app_module, "CHECKPOINT_DIR", self.checkpoints),
        ]
        for current_patch in self.patches:
            current_patch.start()

        app = app_module.create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def tearDown(self):
        for current_patch in reversed(self.patches):
            current_patch.stop()
        self.temp_dir.cleanup()

    def upload_image(self, filename="sheep.jpg"):
        return self.client.post(
            "/upload",
            data={"files[]": (io.BytesIO(b"fake image"), filename)},
            content_type="multipart/form-data",
        )

    def make_checkpoint(self, name="yolo11s.pt"):
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        (self.checkpoints / name).write_bytes(b"fake weights")

    def fake_prediction(self, args):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "sheep.jpg").write_bytes(b"annotated image")
        csv_path = self.run_dir / "prediction_counts.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["image", "sheep", "cattle"]
            )
            writer.writeheader()
            writer.writerow({"image": "sheep.jpg", "sheep": 4, "cattle": 1})
        return [SimpleNamespace(save_dir=self.run_dir)], csv_path

    def test_index_renders_the_browser_interface(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"YOLO Livestock Detection", response.data)
        self.assertIn(b"yolo11s.pt", response.data)

    def test_upload_saves_a_safe_filename(self):
        response = self.upload_image("../sheep.jpg")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["uploaded"], ["sheep.jpg"])
        self.assertTrue((self.uploads / "sheep.jpg").is_file())

    def test_upload_rejects_an_invalid_file_type(self):
        response = self.upload_image("notes.txt")

        self.assertEqual(response.status_code, 400)
        self.assertIn("No valid images", response.get_json()["error"])

    def test_prediction_requires_uploaded_images(self):
        response = self.client.post(
            "/predict", json={"model": "yolo11s.pt", "files": []}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("No uploaded images", response.get_json()["error"])

    def test_prediction_rejects_unknown_models(self):
        response = self.client.post(
            "/predict", json={"model": "unknown.pt", "files": []}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown model", response.get_json()["error"])

    def test_prediction_rejects_invalid_thresholds(self):
        self.upload_image()
        self.make_checkpoint()
        response = self.client.post(
            "/predict",
            json={
                "model": "yolo11s.pt",
                "files": ["sheep.jpg"],
                "conf": 2,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("between 0 and 1", response.get_json()["error"])

    def test_prediction_returns_images_and_count_metrics(self):
        self.upload_image()
        self.make_checkpoint()

        with patch.object(
            app_module, "predict_model", side_effect=self.fake_prediction
        ):
            response = self.client.post(
                "/predict",
                json={"model": "yolo11s.pt", "files": ["sheep.jpg"]},
            )

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["success"])
        self.assertEqual(data["images"], ["/static/results/sheep.jpg"])
        self.assertEqual(data["metrics"]["num_images"], 1)
        self.assertEqual(data["metrics"]["total_detections"], 5)
        self.assertTrue((self.results / "sheep.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
