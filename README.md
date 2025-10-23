# YOLO Livestock Detection

This project detects and counts livestock in images and videos with Ultralytics YOLO. It includes dataset preparation, training, validation, prediction, count reports and a browser interface.

## Project structure

```text
checkpoints/                         local model weights for the browser
demo_yolo/
  data_preprocessing/
    cleanup_yolo_unlabeled.py       remove images with no labels
    extract_frames.py               extract frames from videos
    png_to_jpeg.py                  convert image folders to JPEG
  tests/
    test_yolo_pipeline.py           model pipeline tests
  merge_to_yolo.py                  merge and split labelled datasets
  yolo_pipeline.py                  train, validate and predict
gui/
  static/app.js                     browser behaviour
  templates/index.html              browser page
  app.py                            Flask server
gui_test/
  test_app.py                       Flask and interface tests
requirements.txt
```

Datasets, generated runs, model weights and private project documents are excluded from Git.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The commands use CPU by default. Pass `--device 0` to use the first CUDA GPU.

## Merge datasets

Each source needs an `images` folder and a matching `labels` folder. The folders can be flat or already contain train, validation and test splits.

YOLO text labels are expected inside the labels folder. LabelMe JSON labels can be stored in the labels folder or beside each image.

Create `classes.txt` with one class name per line:

```text
sheep
cattle
```

Merge and split the sources:

```bash
python3 demo_yolo/merge_to_yolo.py \
  --classes classes.txt \
  --out merged_ds \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  raw_dataset/source_one raw_dataset/source_two
```

The script checks labels, converts LabelMe boxes, avoids filename overwrites and creates `merged_ds/dataset.yaml`. Seed 42 is used by default so the same split can be repeated.

## Clean a dataset

Preview images with missing or empty labels:

```bash
python3 demo_yolo/data_preprocessing/cleanup_yolo_unlabeled.py \
  merged_ds \
  --dry-run
```

Run without `--dry-run` to move rejected files into `merged_ds/no_label`. Use `--mode delete` only when those files are no longer needed.

## Prepare images and videos

Extract every tenth frame from a video:

```bash
python3 demo_yolo/data_preprocessing/extract_frames.py video.mp4 \
  --out frames \
  --every 10
```

Convert an image folder to JPEG while keeping its folder structure:

```bash
python3 demo_yolo/data_preprocessing/png_to_jpeg.py raw_images \
  --out jpeg_images \
  --quality 85
```

## Train a model

```bash
python3 demo_yolo/yolo_pipeline.py train \
  --model yolo11n.pt \
  --data merged_ds/dataset.yaml \
  --epochs 50 \
  --run-name livestock_model \
  --device cpu
```

The best weights are saved under `runs/detect/livestock_model/weights`.

## Validate a model

Use a training run name:

```bash
python3 demo_yolo/yolo_pipeline.py val \
  --run-name livestock_model \
  --data merged_ds/dataset.yaml \
  --split test \
  --device cpu
```

You can also pass a checkpoint directly with `--weights path/to/best.pt`. Validation prints precision, recall, mAP50 and mAP50-95.

## Run predictions

The source can be an image, video or folder:

```bash
python3 demo_yolo/yolo_pipeline.py predict \
  --run-name livestock_model \
  --source path/to/images \
  --output-name livestock_results \
  --device cpu
```

The output folder contains annotated files and `prediction_counts.csv`. The CSV stores the number of each livestock class detected in every image.

## Browser interface

The browser supports YOLO 8, YOLO 10 and YOLO 11 model choices. Put the required `.pt` files in `checkpoints/` before starting it. See [checkpoints/README.md](checkpoints/README.md) for the accepted filenames.

Start the Flask server:

```bash
python3 -m gui.app
```

Open `http://127.0.0.1:5002`. Upload images, choose a model, adjust confidence and IoU, then run the prediction. Annotated images and count summaries are shown in the page.

## Tests

Run all tests:

```bash
python3 -m pytest
```

The tests use fake model results and temporary folders. They do not download weights or start model training.

Run coverage if needed:

```bash
python3 -m pytest --cov=demo_yolo --cov=gui --cov-report=term-missing
```
