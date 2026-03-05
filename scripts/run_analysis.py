"""CLI for golf swing analysis pipeline.

Usage:
    python scripts/run_analysis.py --golfer ymirza --day oct25 --phases backswing downswing
    python scripts/run_analysis.py --golfer ymirza --no-filter-sig
"""
import argparse
import os
import sys

from eagle_swing.analysis.core import (
    build_swing_items,
    load_golfer_data,
    resolve_group_by_score,
    perform_group_analysis,
    ALL_METRICS,
    COCO_BODY,
    RESAMPLE_TARGET,
)


def run(args):
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else None
    if data_dir is None:
        from eagle_swing.data.loader import DATA_ROOT
        data_dir = os.path.dirname(DATA_ROOT)

    scored = os.path.join(data_dir, f"{args.golfer}_scored_lbls.csv")
    fallback = os.path.join(data_dir, f"{args.golfer}_lbls.csv")
    csv_path = scored if os.path.isfile(scored) else fallback
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)
    print(f"Using CSV: {os.path.basename(csv_path)}")

    out_dir = os.path.abspath(args.out or f"{args.golfer}_analysis_output")
    os.makedirs(out_dir, exist_ok=True)

    day_filter = args.day if args.day else None

    print(f"Loading {args.golfer} data from {csv_path} ...")
    df = load_golfer_data(
        csv_path, args.golfer, data_dir,
        back_frames=args.back_frames, down_frames=args.down_frames,
        day_filter=day_filter,
    )
    print(f"  {len(df)} swings loaded")

    items = build_swing_items(df)
    print(f"  {len(items)} SwingData objects built")

    src_pool = resolve_group_by_score(items, args.src_score)
    tgt_pool = resolve_group_by_score(items, args.tgt_score)
    print(f"  Source (score={args.src_score}): {len(src_pool)} swings")
    print(f"  Target (score={args.tgt_score}): {len(tgt_pool)} swings")

    if not src_pool or not tgt_pool:
        print("ERROR: Need at least one swing in each group.")
        sys.exit(1)

    for phase in args.phases:
        print(f"\n--- {phase} analysis ---")
        results = perform_group_analysis(
            src_pool, tgt_pool, ALL_METRICS,
            mode="1d", phase=phase, target_frames=RESAMPLE_TARGET,
        )
        sig_metrics = [k for k, v in results.items() if v["sig"].any()]
        print(f"  Significant metrics: {len(sig_metrics)}/{len(ALL_METRICS)}")
        for m in sig_metrics:
            print(f"    - {m}")

    print(f"\nOutputs in: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Golf swing analysis pipeline")
    ap.add_argument("--golfer", default="ymirza", help="Golfer name")
    ap.add_argument("--data-dir", default=None, help="Directory with CSV and pkl files")
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--src-score", type=int, default=1, help="Source group score (best)")
    ap.add_argument("--tgt-score", type=int, default=5, help="Target group score (worst)")
    ap.add_argument("--phases", nargs="+", default=["backswing", "downswing"])
    ap.add_argument("--back-frames", type=int, default=60)
    ap.add_argument("--down-frames", type=int, default=30)
    ap.add_argument("--day", nargs="+", default=None, help="Filter by day(s)")
    ap.add_argument("--no-filter-sig", action="store_true",
                    help="Show all metrics, not just significant")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
