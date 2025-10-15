"""Extract JPEG frames from a video."""

import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, every=1, quality=90):
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_number = 0
    saved = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break

        if frame_number % every == 0:
            output_path = output_dir / f"frame_{frame_number:06d}.jpg"
            cv2.imwrite(
                str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            saved += 1
        frame_number += 1

    video.release()
    return saved


def build_parser():
    parser = argparse.ArgumentParser(description="Extract JPEG frames from a video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("frames"))
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--quality", type=int, default=90)
    return parser


def main():
    args = build_parser().parse_args()
    if args.every < 1:
        raise SystemExit("--every must be at least 1")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")

    try:
        saved = extract_frames(args.video, args.out, args.every, args.quality)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"[INFO] Saved {saved} frames to {args.out}")


if __name__ == "__main__":
    main()
