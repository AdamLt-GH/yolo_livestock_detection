"""Merge a few YOLO datasets into one folder."""

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
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


def merge_datasets(sources, output_dir):
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    used_names = {path.name.lower() for path in images_out.glob("*")}
    copied = 0

    for source in sources:
        pairs = find_pairs(source)
        print(f"[INFO] Found {len(pairs)} labelled images in {source}")

        for image_path, label_path in pairs:
            output_name = next_name(image_path, used_names)
            output_label = f"{Path(output_name).stem}.txt"

            shutil.copy2(image_path, images_out / output_name)
            shutil.copy2(label_path, labels_out / output_label)
            copied += 1

    return copied


def build_parser():
    parser = argparse.ArgumentParser(
        description="Merge YOLO datasets that contain images and labels folders"
    )
    parser.add_argument("--out", required=True, help="Output dataset folder")
    parser.add_argument("sources", nargs="+", help="Dataset folders to merge")
    return parser


def main():
    args = build_parser().parse_args()
    sources = [Path(source) for source in args.sources]
    copied = merge_datasets(sources, Path(args.out))

    if copied == 0:
        raise SystemExit("No labelled images were copied")

    print(f"[INFO] Finished merging {copied} labelled images")


if __name__ == "__main__":
    main()
