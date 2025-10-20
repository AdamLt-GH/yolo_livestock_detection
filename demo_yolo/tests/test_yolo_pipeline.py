import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from demo_yolo import yolo_pipeline


class FakeResult:
    def __init__(self, image_path, save_dir, class_ids):
        self.path = image_path
        self.save_dir = save_dir
        self.boxes = SimpleNamespace(
            cls=SimpleNamespace(tolist=lambda: class_ids)
        )


class FakeModel:
    names = {0: "sheep", 1: "cattle"}

    def __init__(self, model_path, save_dir):
        self.model_path = model_path
        self.save_dir = save_dir
        self.train_options = None
        self.val_options = None
        self.predict_options = None

    def train(self, **options):
        self.train_options = options
        return SimpleNamespace(save_dir=self.save_dir)

    def val(self, **options):
        self.val_options = options
        return SimpleNamespace(
            save_dir=self.save_dir,
            results_dict={
                "metrics/precision(B)": 0.91,
                "metrics/recall(B)": 0.82,
                "metrics/mAP50(B)": 0.76,
                "metrics/mAP50-95(B)": 0.53,
            },
        )

    def predict(self, **options):
        self.predict_options = options
        return [
            FakeResult("images/b.jpg", self.save_dir, [1]),
            FakeResult("images/a.jpg", self.save_dir, [0, 0]),
        ]


class YoloPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = Path(self.temp_dir.name) / "runs" / "detect" / "test"
        self.model = None

        def fake_loader(model_path):
            self.model = FakeModel(model_path, self.save_dir)
            return self.model

        self.loader_patch = patch.object(
            yolo_pipeline, "load_model", side_effect=fake_loader
        )
        self.loader_patch.start()

    def tearDown(self):
        self.loader_patch.stop()
        self.temp_dir.cleanup()

    def test_parser_has_train_validation_and_prediction(self):
        parser = yolo_pipeline.build_parser()
        train = parser.parse_args(["train"])
        validate = parser.parse_args(["val"])
        predict = parser.parse_args(["predict", "--source", "images"])

        self.assertIs(train.func, yolo_pipeline.train_model)
        self.assertIs(validate.func, yolo_pipeline.validate_model)
        self.assertIs(predict.func, yolo_pipeline.predict_model)

    def test_training_passes_the_selected_options(self):
        args = Namespace(
            model="yolo11n.pt",
            data="dataset.yaml",
            epochs=5,
            imgsz=640,
            batch=8,
            device="cpu",
            workers=2,
            seed=42,
            run_name="livestock",
        )
        yolo_pipeline.train_model(args)

        self.assertEqual(self.model.model_path, "yolo11n.pt")
        self.assertEqual(self.model.train_options["epochs"], 5)
        self.assertEqual(self.model.train_options["name"], "livestock")

    def test_validation_finds_weights_from_the_run_name(self):
        args = Namespace(
            weights=None,
            run_name="livestock",
            data="dataset.yaml",
            split="test",
            device="cpu",
        )
        yolo_pipeline.validate_model(args)

        self.assertEqual(
            self.model.model_path,
            "runs/detect/livestock/weights/best.pt",
        )
        self.assertEqual(self.model.val_options["split"], "test")

    def test_prediction_writes_sorted_livestock_counts(self):
        args = Namespace(
            source="images",
            weights="weights.pt",
            run_name="train",
            output_name="results",
            conf=0.25,
            iou=0.7,
            imgsz=640,
            device="cpu",
            save=True,
            save_txt=False,
        )
        results, csv_path = yolo_pipeline.predict_model(args)

        self.assertEqual(len(results), 2)
        self.assertEqual(self.model.predict_options["name"], "results")
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual(
            rows,
            [
                {"image": "a.jpg", "sheep": "2", "cattle": "0"},
                {"image": "b.jpg", "sheep": "0", "cattle": "1"},
            ],
        )

    def test_empty_predictions_do_not_create_a_csv(self):
        self.model = FakeModel("weights.pt", self.save_dir)
        self.model.predict = lambda **options: []

        with patch.object(yolo_pipeline, "load_model", return_value=self.model):
            args = Namespace(
                source="images",
                weights="weights.pt",
                run_name="train",
                output_name="results",
                conf=0.25,
                iou=0.7,
                imgsz=640,
                device="cpu",
                save=True,
                save_txt=False,
            )
            results, csv_path = yolo_pipeline.predict_model(args)

        self.assertEqual(results, [])
        self.assertIsNone(csv_path)


if __name__ == "__main__":
    unittest.main()
