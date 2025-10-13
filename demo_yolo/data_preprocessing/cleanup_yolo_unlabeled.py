"""Find images that have no YOLO labels and move or delete them."""

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "valid", "test")


def has_labels(label_path):
    if not label_path.is_file():
        return False
    return any(
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
    )


def find_dataset_splits(dataset_root):
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError("The dataset needs images and labels folders")

    found = [
        split
        for split in SPLITS
        if (images_dir / split).is_dir() and (labels_dir / split).is_dir()
    ]
    return found or [""]


def clean_dataset(dataset_root, mode="move", destination=None, dry_run=False):
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    destination = destination or dataset_root / "no_label"
    scanned = 0
    removed = 0

    for split in find_dataset_splits(dataset_root):
        split_images = images_dir / split
        split_labels = labels_dir / split

        for image_path in sorted(split_images.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            scanned += 1
            relative_path = image_path.relative_to(split_images)
            label_path = (split_labels / relative_path).with_suffix(".txt")
            if has_labels(label_path):
                continue

            removed += 1
            print(f"[{mode.upper()}] {image_path}")
            if dry_run:
                continue

            if mode == "delete":
                image_path.unlink(missing_ok=True)
                label_path.unlink(missing_ok=True)
            else:
                image_target = destination / "images" / split / relative_path
                label_target = (
                    destination / "labels" / split / relative_path
                ).with_suffix(".txt")
                image_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image_path), str(image_target))
                if label_path.exists():
                    label_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(label_path), str(label_target))

    return scanned, removed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Clean images with missing or empty YOLO labels"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--mode", choices=["move", "delete"], default="move")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        scanned, removed = clean_dataset(
            args.dataset_root.resolve(), args.mode, args.dest, args.dry_run
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    action = "would be removed" if args.dry_run else "removed"
    print(f"[INFO] Scanned {scanned} images and {removed} {action}")


if __name__ == "__main__":
    main()
