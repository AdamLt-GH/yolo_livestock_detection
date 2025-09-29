"""Merge a few YOLO datasets into one folder."""

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def find_pairs(source):
    images_dir = source / "images"
    labels_dir = source / "labels"
    pairs = []

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[WARN] Could not find images and labels in {source}")
        return pairs

    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        text_label = labels_dir / f"{image_path.stem}.txt"
        json_label = labels_dir / f"{image_path.stem}.json"
        nearby_json = image_path.with_suffix(".json")

        if text_label.exists():
            pairs.append((image_path, text_label))
        elif json_label.exists():
            pairs.append((image_path, json_label))
        elif nearby_json.exists():
            pairs.append((image_path, nearby_json))
        else:
            print(f"[WARN] No label found for {image_path.name}")

    return pairs


def next_name(image_path, used_names):
    name = image_path.name
    number = 2

    # Some datasets use the same file names so do not overwrite them
    while name.lower() in used_names:
        name = f"{image_path.stem}_{number}{image_path.suffix.lower()}"
        number += 1

    used_names.add(name.lower())
    return name


def load_classes(classes_path):
    names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError("The classes file is empty")
    return names


def clean_labels(label_path, max_class_id):
    cleaned = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            continue

        if not 0 <= class_id <= max_class_id:
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1):
            continue
        if not (0 < width <= 1 and 0 < height <= 1):
            continue

        cleaned.append(
            f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"
        )

    return cleaned


def convert_labelme(label_path, class_map, image_width, image_height):
    data = json.loads(label_path.read_text(encoding="utf-8"))
    labels = []

    for shape in data.get("shapes", []):
        class_name = (shape.get("label") or "").strip()
        points = shape.get("points") or []
        if class_name not in class_map or not points:
            continue

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, x2 = max(0, min(xs)), min(image_width, max(xs))
        y1, y2 = max(0, min(ys)), min(image_height, max(ys))
        if x2 <= x1 or y2 <= y1:
            continue

        x = ((x1 + x2) / 2) / image_width
        y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        labels.append(
            f"{class_map[class_name]} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"
        )

    return labels


def write_dataset_yaml(output_dir, class_names):
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names: [" + ", ".join(json.dumps(name) for name in class_names) + "]",
    ]
    (output_dir / "dataset.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def split_pairs(pairs, val_ratio, test_ratio, seed):
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    test_count = int(len(shuffled) * test_ratio)
    return {
        "val": shuffled[:val_count],
        "test": shuffled[val_count : val_count + test_count],
        "train": shuffled[val_count + test_count :],
    }


def merge_datasets(sources, output_dir, class_names, val_ratio, test_ratio, seed):
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_pairs = []
    copied = 0
    class_map = {name: number for number, name in enumerate(class_names)}

    for source in sources:
        pairs = find_pairs(source)
        print(f"[INFO] Found {len(pairs)} labelled images in {source}")
        all_pairs.extend(pairs)

    for split, pairs in split_pairs(all_pairs, val_ratio, test_ratio, seed).items():
        images_out = output_dir / "images" / split
        labels_out = output_dir / "labels" / split
        used_names = {path.name.lower() for path in images_out.glob("*")}

        for image_path, label_path in pairs:
            if label_path.suffix.lower() == ".txt":
                labels = clean_labels(label_path, len(class_names) - 1)
            else:
                try:
                    from PIL import Image

                    with Image.open(image_path) as image:
                        width, height = image.size
                    labels = convert_labelme(label_path, class_map, width, height)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    print(f"[WARN] Could not convert {label_path}: {error}")
                    continue

            if not labels:
                print(f"[WARN] No valid labels in {label_path}")
                continue

            output_name = next_name(image_path, used_names)
            output_label = f"{Path(output_name).stem}.txt"

            shutil.copy2(image_path, images_out / output_name)
            (labels_out / output_label).write_text(
                "\n".join(labels) + "\n", encoding="utf-8"
            )
            copied += 1

    return copied


def build_parser():
    parser = argparse.ArgumentParser(
        description="Merge YOLO datasets that contain images and labels folders"
    )
    parser.add_argument("--classes", required=True, help="File with one class per line")
    parser.add_argument("--out", required=True, help="Output dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("sources", nargs="+", help="Dataset folders to merge")
    return parser


def main():
    args = build_parser().parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise SystemExit("Split ratios cannot be negative")
    if args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("Validation and test ratios must add up to less than 1")

    class_names = load_classes(Path(args.classes))
    sources = [Path(source) for source in args.sources]
    copied = merge_datasets(
        sources,
        Path(args.out),
        class_names,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    if copied == 0:
        raise SystemExit("No labelled images were copied")

    write_dataset_yaml(Path(args.out), class_names)
    print(f"[INFO] Finished merging {copied} labelled images")


if __name__ == "__main__":
    main()
