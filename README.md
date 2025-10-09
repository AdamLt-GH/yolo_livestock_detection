# YOLO Livestock Detection

This project will use YOLO to detect and count livestock in images and videos.

The main goals are:

- prepare labelled livestock datasets
- train and validate YOLO models
- run predictions on images and videos
- save the number of detected animals

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the requirements:

```bash
python3 -m pip install -r requirements.txt
```

## Merge datasets

The first dataset tool combines YOLO datasets that have matching `images` and `labels` folders:

```bash
python3 demo_yolo/merge_to_yolo.py \
  --classes classes.txt \
  --out merged_ds \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  raw_dataset/source_one raw_dataset/source_two
```

The classes file contains one livestock class name per line. The script supports YOLO text labels and LabelMe JSON labels, checks class IDs against the list and creates `dataset.yaml` for training. It also creates separate train, validation and test folders. The random split uses seed 42 by default so it can be repeated.

## Train a model

The training command uses CPU by default. Use `--device 0` to train with the first CUDA GPU.

```bash
python3 demo_yolo/yolo_pipeline.py train \
  --model yolo11n.pt \
  --data merged_ds/dataset.yaml \
  --epochs 50 \
  --run-name livestock_model \
  --device cpu
```

The trained weights are saved under `runs/detect/livestock_model/weights`.

## Validate a model

Pass the saved weights directly or use the name of an earlier training run:

```bash
python3 demo_yolo/yolo_pipeline.py val \
  --run-name livestock_model \
  --data merged_ds/dataset.yaml \
  --split test \
  --device cpu
```

The command prints precision, recall and mAP results after validation.
