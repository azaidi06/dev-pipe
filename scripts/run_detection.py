"""Batch backswing + contact detection CLI.

Usage:
    python scripts/run_detection.py <dataset_dir> [options]
    python scripts/run_detection.py /path/to/ymirza/oct25 --contact --csv
"""
import argparse
import csv
import os
import sys

import numpy as np

from eagle_swing.detection.backswing import detect_backswings
from eagle_swing.detection.contact import detect_contacts
from eagle_swing.detection.config import Config


def discover_videos(dataset_dir, skip=None):
    """Find paired (pkl, mov) files in a dataset directory."""
    skip = set(skip or [])
    videos = {}
    for entry in sorted(os.listdir(dataset_dir)):
        if entry in skip:
            continue
        pkl = os.path.join(dataset_dir, entry, "keypoints", entry + ".pkl")
        mov = os.path.join(dataset_dir, entry + ".MOV")
        if os.path.isfile(pkl) and os.path.isfile(mov):
            videos[entry] = {"mov": mov, "pkl": pkl}
    return videos


def flag_problems(result, cfg):
    """Flag potential issues in detection results."""
    issues = []
    pf = result.peak_frames
    smoothed = result.smoothed
    pkl_data = result.pkl_data
    fps = result.fps
    n = len(pf)

    if n < cfg.min_expected_swings:
        issues.append((None, f"Only {n} swing(s) (expected >= {cfg.min_expected_swings})"))
    if n > cfg.max_expected_swings:
        issues.append((None, f"{n} swings (expected <= {cfg.max_expected_swings})"))
    if n == 0:
        return issues

    vals = smoothed[pf]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) if len(vals) > 1 else 0.0

    for i, p in enumerate(pf):
        if mad > 0 and len(vals) >= 3:
            z = abs(vals[i] - med) / mad
            if z > cfg.flag_mad_threshold:
                issues.append((i, f"x+y={vals[i]:.0f} is {z:.1f} MADs from median"))
        s = max(0, p - cfg.low_conf_window)
        e = min(len(pkl_data), p + cfg.low_conf_window + 1)
        confs = [
            (pkl_data[f"frame_{f}"]["keypoint_scores"][cfg.left_wrist]
             + pkl_data[f"frame_{f}"]["keypoint_scores"][cfg.right_wrist]) / 2.0
            for f in range(s, e) if f"frame_{f}" in pkl_data
        ]
        if confs and np.mean(confs) < cfg.low_conf_threshold:
            issues.append((i, f"Low wrist conf: {np.mean(confs):.2f}"))
        if i > 0:
            gap_s = (p - pf[i - 1]) / fps
            if gap_s < cfg.close_gap_seconds:
                issues.append((i, f"Only {gap_s:.1f}s since previous swing"))
    return issues


def export_csvs(all_bs, all_ct, out_dir):
    """Export detection results to CSV."""
    contact_lookup = {}
    for cr in all_ct:
        br = cr.backswing_result
        for i, cf in enumerate(cr.contact_frames):
            if cf >= 0 and i < len(br.peak_frames):
                bf = br.peak_frames[i]
                gap = int(cf - bf)
                contact_lookup[(cr.name, i + 1)] = dict(
                    contact_frame=int(cf), contact_time_s=round(cf / br.fps, 2),
                    downswing_frames=gap, downswing_time_s=round(gap / br.fps, 3),
                )

    rows = []
    for br in all_bs:
        for i, bf in enumerate(br.peak_frames):
            row = dict(
                video=br.name, swing_num=i + 1,
                backswing_frame=int(bf), backswing_time_s=round(bf / br.fps, 2),
                contact_frame="", contact_time_s="",
                downswing_frames="", downswing_time_s="",
                fps=round(br.fps, 2),
            )
            ct = contact_lookup.get((br.name, i + 1))
            if ct:
                row.update(ct)
            rows.append(row)

    fields = [
        "video", "swing_num", "backswing_frame", "backswing_time_s",
        "contact_frame", "contact_time_s", "downswing_frames", "downswing_time_s", "fps",
    ]
    csv_path = os.path.join(out_dir, "swings.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {csv_path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Batch backswing + contact detection")
    ap.add_argument("dataset_dir", help="Directory with video subdirectories")
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--contact", action="store_true", help="Also detect contacts")
    ap.add_argument("--csv", action="store_true", help="Export results CSV")
    ap.add_argument("--skip", action="append", default=[], help="Skip specific videos")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    dataset_name = os.path.basename(dataset_dir)
    out_root = os.path.abspath(args.out or (dataset_name + "_detection_output"))
    os.makedirs(out_root, exist_ok=True)
    cfg = Config()

    videos = discover_videos(dataset_dir, skip=args.skip)
    print(f"Found {len(videos)} videos: {', '.join(videos.keys())}")

    all_bs, all_ct, all_problems = [], [], {}
    for name, paths in videos.items():
        result = detect_backswings(paths["pkl"], paths["mov"], config=cfg)
        all_bs.append(result)
        line = f"{name}: {result.n_swings} swings"

        ct = None
        if args.contact:
            ct = detect_contacts(result, config=cfg)
            all_ct.append(ct)
            line += f", {ct.n_contacts} contacts"

        filters = [m.split(":")[0] for m in result.filter_log]
        if filters:
            line += f"  [filters: {', '.join(filters)}]"

        issues = flag_problems(result, cfg)
        if issues:
            all_problems[name] = issues
            line += " *** PROBLEMS ***"
        print(line)

    txt = f"\n{dataset_name}: {sum(r.n_swings for r in all_bs)} total swings across {len(all_bs)} videos"
    if all_ct:
        txt += f", {sum(c.n_contacts for c in all_ct)} contacts"
    if all_problems:
        txt += f"\nProblematic: {', '.join(all_problems.keys())}"
        summary_path = os.path.join(out_root, "problems.txt")
        with open(summary_path, "w") as f:
            for vname, issues in all_problems.items():
                f.write(f"{vname}:\n")
                for si, reason in issues:
                    f.write(f"  {'Swing ' + str(si + 1) if si is not None else 'VIDEO'}: {reason}\n")
                f.write("\n")
    print(txt)

    if args.csv:
        export_csvs(all_bs, all_ct, out_root)
    print(f"All outputs in: {out_root}")


if __name__ == "__main__":
    main()
