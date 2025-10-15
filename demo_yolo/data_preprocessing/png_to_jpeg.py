"""Convert dataset images to JPEG while keeping the folder structure."""

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def convert_images(input_dir, output_dir, quality=85, delete_original=False):
    from PIL import Image

    converted = 0
    for input_path in sorted(input_dir.rglob("*")):
        if not input_path.is_file():
            continue
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = input_path.relative_to(input_dir).with_suffix(".jpg")
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(input_path) as image:
                # JPEG cannot keep transparency so RGB is simpler here
                image.convert("RGB").save(output_path, "JPEG", quality=quality)
        except OSError as error:
            print(f"[WARN] Could not convert {input_path}: {error}")
            continue

        if delete_original and input_path.resolve() != output_path.resolve():
            input_path.unlink()
        converted += 1

    return converted


def build_parser():
    parser = argparse.ArgumentParser(description="Convert dataset images to JPEG")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--delete-original", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {args.input_dir}")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")

    output_dir = args.out or args.input_dir
    converted = convert_images(
        args.input_dir, output_dir, args.quality, args.delete_original
    )
    print(f"[INFO] Converted {converted} images into {output_dir}")


if __name__ == "__main__":
    main()
