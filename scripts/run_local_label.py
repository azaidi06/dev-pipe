"""Local video labeling CLI wrapper.

This script wraps the pipeline's label_videos worker for local usage.
Requires the full pipeline environment (mmpose, torch, opencv).

Usage:
    python scripts/run_local_label.py /path/to/videos
    python scripts/run_local_label.py /path/to/videos --workers 3 --skip-existing
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Local video labeling (ViTPose-Huge)")
    ap.add_argument("video_dir", help="Directory with video files or single video path")
    ap.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers")
    ap.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    ap.add_argument("--skip-existing", action="store_true", help="Skip already-labeled videos")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    try:
        from eagle_swing.pipeline.label import label_local
        label_local(args.video_dir, skip_existing=args.skip_existing, batch_size=args.batch_size)
    except NotImplementedError as e:
        print(f"NOTE: {e}")
        print("\nTo run labeling directly, use the pipeline worker:")
        print(f"  python -m label_videos.worker --local {args.video_dir} "
              f"-w {args.workers} --gpu {args.gpu}")
        if args.skip_existing:
            print("  (add --skip-existing)")
        sys.exit(1)


if __name__ == "__main__":
    main()
