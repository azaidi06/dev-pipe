#!/usr/bin/env python3
"""Generate JSON data for the Refinement Explorer React page.

Processes all ymirza pkl files through both production (1D) and ensemble (2D)
detection, producing a JSON file consumed by the docs site.

Usage:
    python scripts/generate_refinement_data.py [--output docs/public/data/refinement_data.json]
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

# Add parent to path so eagle_swing is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eagle_swing.detection.backswing import detect_backswings
from eagle_swing.detection.contact import detect_contacts
from eagle_swing.detection.config import Config
from eagle_swing.detection.refine import (
    get_all_method_picks,
    refine_backswing_apex,
    refine_contact_point,
    validate_peaks_with_score_hand,
    VOTING_STRATEGIES,
)

DATA_ROOT = os.environ.get(
    "EAGLE_SWING_DATA",
    "/home/azaidi/Desktop/golf/data/full_videos/ymirza",
)

# Refinement window sizes (frames before/after the production peak)
# Store wide windows; the React UI crops client-side via sliders
BS_BEFORE = 40
BS_AFTER = 50
CT_OFFSET_MIN = 5
CT_OFFSET_MAX = 70


def find_pkl_files(root):
    """Discover all pkl files under the data root."""
    root = Path(root)
    pkls = []
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith('.'):
            continue
        session = session_dir.name
        for video_dir in sorted(session_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            pkl_path = video_dir / "keypoints" / f"{video_dir.name}.pkl"
            mov_path = video_dir / f"{video_dir.name}.mp4"
            if pkl_path.exists():
                pkls.append({
                    'session': session,
                    'video': video_dir.name,
                    'pkl_path': str(pkl_path),
                    'mov_path': str(mov_path) if mov_path.exists() else None,
                })
    return pkls


def extract_wrist_trajectory(pkl_data, start, end, wrist_idx=10):
    """Extract right wrist (x, y) for a frame range from raw pkl data."""
    n = sum(1 for k in pkl_data if isinstance(k, str) and k.startswith("frame_"))
    start = max(0, start)
    end = min(end, n)
    xs, ys = [], []
    for i in range(start, end):
        key = f"frame_{i}"
        if key in pkl_data:
            kp = pkl_data[key]["keypoints"]
            xs.append(float(kp[wrist_idx][0]))
            ys.append(float(kp[wrist_idx][1]))
        else:
            xs.append(0.0)
            ys.append(0.0)
    return np.array(xs), np.array(ys)


def _count_pkl_frames(pkl_path):
    """Count frame_N keys in a pkl file without loading all data."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return sum(1 for k in data if isinstance(k, str) and k.startswith("frame_"))


def process_one_video(info):
    """Run detection + ensemble refinement for one video, return swing data."""
    pkl_path = info['pkl_path']
    mov_path = info['mov_path']

    # If no video and no __meta__, supply defaults
    n_pkl = _count_pkl_frames(pkl_path)
    default_fps = 60.0

    try:
        result = detect_backswings(
            pkl_path, mov_path,
            fps=default_fps, total_frames=n_pkl,
        )
    except Exception as e:
        print(f"  SKIP {info['video']}: {e}")
        return []

    if result.n_swings == 0:
        return []

    # Extract full keypoints for score-hand validation
    pkl_data = result.pkl_data
    n_frames = sum(1 for k in pkl_data if isinstance(k, str) and k.startswith("frame_"))
    all_kps = np.array([pkl_data[f"frame_{i}"]["keypoints"] for i in range(n_frames)])
    all_scores = np.array([pkl_data[f"frame_{i}"]["keypoint_scores"] for i in range(n_frames)])
    validated = validate_peaks_with_score_hand(result.peak_frames, all_kps, all_scores)

    contact_result = detect_contacts(result)
    swings = []

    for si, peak in enumerate(result.peak_frames):
        peak = int(peak)
        # Backswing refinement window
        bs_start = max(0, peak - BS_BEFORE)
        bs_end = peak + BS_AFTER
        rw_x, rw_y = extract_wrist_trajectory(result.pkl_data, bs_start, bs_end)
        if len(rw_x) < 5:
            continue

        wrist_xy = np.column_stack([rw_x, rw_y])
        prod_rel = peak - bs_start  # production pick relative to window

        # Run all backswing methods
        bs_picks = get_all_method_picks(wrist_xy, direction='backswing')

        # Run all voting strategies
        bs_voting = {}
        for vote_name in VOTING_STRATEGIES:
            r = refine_backswing_apex(wrist_xy, voting=vote_name)
            bs_voting[vote_name] = r['ensemble_idx']

        # Contact window
        ct_start = peak + CT_OFFSET_MIN
        ct_end = peak + CT_OFFSET_MAX
        ct_x, ct_y = extract_wrist_trajectory(result.pkl_data, ct_start, ct_end)
        ct_picks = {}
        ct_voting = {}
        ct_prod_rel = None

        if len(ct_x) >= 5:
            ct_xy = np.column_stack([ct_x, ct_y])
            ct_picks = get_all_method_picks(ct_xy, direction='contact')
            for vote_name in VOTING_STRATEGIES:
                r = refine_contact_point(ct_xy, voting=vote_name)
                ct_voting[vote_name] = r['ensemble_idx']
            # Production contact pick
            cf = int(contact_result.contact_frames[si])
            if cf >= 0:
                ct_prod_rel = cf - ct_start

        # Full skeleton keypoints for frame scrubber (±10 around landmark)
        def _make_skeleton(center_frame):
            skel_before, skel_after = 10, 10
            s = max(0, center_frame - skel_before)
            e = min(n_frames, center_frame + skel_after + 1)
            return {
                'start_frame': s,
                'peak_rel': center_frame - s,
                'keypoints': np.round(all_kps[s:e], 1).tolist(),
                'scores': np.round(all_scores[s:e], 2).tolist(),
            }

        skeleton_data = _make_skeleton(peak)

        # Contact skeleton (centered on production contact frame)
        cf = int(contact_result.contact_frames[si])
        contact_skeleton_data = _make_skeleton(cf) if cf >= 0 else None

        # 1D signal window for visualization
        sig_start = max(0, peak - 60)
        sig_end = min(len(result.smoothed), peak + 100)
        signal_frames = list(range(sig_start, sig_end))
        signal_smoothed = result.smoothed[sig_start:sig_end].tolist()

        swing_data = {
            'id': f"{info['session']}_{info['video']}_s{si}",
            'session': info['session'],
            'video': info['video'],
            'swing_idx': si,
            'production_frame': peak,
            'fps': float(result.fps) if result.fps else None,
            'validated': bool(validated[si]),
            'backswing': {
                'window_start': bs_start,
                'window_end': bs_start + len(rw_x),
                'wrist_x': rw_x.tolist(),
                'wrist_y': rw_y.tolist(),
                'production_rel': prod_rel,
                'methods': {k: v for k, v in bs_picks.items()},
                'voting': bs_voting,
            },
            'contact': {
                'window_start': ct_start,
                'window_end': ct_start + len(ct_x),
                'wrist_x': ct_x.tolist(),
                'wrist_y': ct_y.tolist(),
                'production_rel': ct_prod_rel,
                'methods': {k: v for k, v in ct_picks.items()},
                'voting': ct_voting,
            },
            'skeleton': skeleton_data,
            'contact_skeleton': contact_skeleton_data,
            'signal': {
                'frames': signal_frames,
                'smoothed': signal_smoothed,
            },
        }
        swings.append(swing_data)

    return swings


def _phase_stats(all_swings, phase_key):
    """Compute prod vs ensemble diffs for a given phase."""
    diffs = []
    for s in all_swings:
        p = s[phase_key]
        prod = p['production_rel']
        ens_mean = p['voting'].get('mean')
        if prod is not None and ens_mean is not None:
            diffs.append(abs(prod - ens_mean))
    if not diffs:
        return {'mean_abs_diff': 0, 'median_abs_diff': 0,
                'pct_agree_1frame': 0, 'pct_agree_3frame': 0, 'pct_differ_5plus': 0}
    return {
        'mean_abs_diff': float(np.mean(diffs)),
        'median_abs_diff': float(np.median(diffs)),
        'pct_agree_1frame': float(np.mean([d <= 1 for d in diffs]) * 100),
        'pct_agree_3frame': float(np.mean([d <= 3 for d in diffs]) * 100),
        'pct_differ_5plus': float(np.mean([d >= 5 for d in diffs]) * 100),
    }


def compute_summary(all_swings):
    """Compute aggregate stats across all swings."""
    sessions = sorted(set(s['session'] for s in all_swings))
    bs_stats = _phase_stats(all_swings, 'backswing')
    ct_stats = _phase_stats(all_swings, 'contact')
    return {
        'total_swings': len(all_swings),
        'total_videos': len(set(s['video'] for s in all_swings)),
        'sessions': sessions,
        **bs_stats,
        'contact': ct_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate refinement explorer data")
    parser.add_argument(
        "--output", default="docs/public/data/refinement_data.json",
        help="Output JSON path",
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="ymirza data root")
    args = parser.parse_args()

    print(f"Scanning {args.data_root} ...")
    pkl_files = find_pkl_files(args.data_root)
    print(f"Found {len(pkl_files)} videos")

    all_swings = []
    for info in pkl_files:
        print(f"  Processing {info['session']}/{info['video']} ...")
        swings = process_one_video(info)
        all_swings.extend(swings)
        print(f"    → {len(swings)} swings")

    summary = compute_summary(all_swings)
    print(f"\nTotal: {summary['total_swings']} swings from {summary['total_videos']} videos")
    print(f"Mean |diff|: {summary['mean_abs_diff']:.1f} frames")
    print(f"Agree ≤1 frame: {summary['pct_agree_1frame']:.0f}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'swings': all_swings, 'summary': summary}, f)
    print(f"\nWrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
