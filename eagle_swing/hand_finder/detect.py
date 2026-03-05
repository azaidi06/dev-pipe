"""Hand-raise detection logic. Pure numpy + scipy, no OpenCV.

Public API:
    find_score_hands(contact_result, config=None) -> HandFinderResult
"""

import numpy as np
from scipy.signal import savgol_filter
from .config import HandFinderConfig, HandFinderResult


def _get_search_windows(contact_result):
    ct = contact_result
    br = ct.backswing_result
    peaks = br.peak_frames
    contacts = ct.contact_frames
    n_pkl = sum(1 for k in br.pkl_data if k.startswith("frame_"))

    windows = []
    for i in range(len(contacts)):
        if contacts[i] == -1:
            windows.append(None)
            continue
        start = int(contacts[i])
        end = int(peaks[i + 1]) if i + 1 < len(peaks) else n_pkl
        windows.append((start, end))
    return windows


def _extract_keypoints_window(pkl_data, start, end):
    frames, scores = [], []
    for i in range(start, end):
        key = f"frame_{i}"
        if key in pkl_data:
            frames.append(pkl_data[key]["keypoints"])
            scores.append(pkl_data[key]["keypoint_scores"])
        else:
            frames.append(np.zeros((17, 2)))
            scores.append(np.zeros(17))
    return np.array(frames), np.array(scores)


def _check_raised(kps, scores, cfg, side):
    if side == "LEFT":
        sh, el, wr = cfg.left_shoulder, cfg.left_elbow, cfg.left_wrist
    else:
        sh, el, wr = cfg.right_shoulder, cfg.right_elbow, cfg.right_wrist
    conf_ok = (scores[:, wr] >= cfg.conf_threshold) & (scores[:, sh] >= cfg.conf_threshold) & (scores[:, el] >= cfg.conf_threshold)
    return conf_ok & (kps[:, wr, 1] < kps[:, sh, 1]) & (kps[:, wr, 1] < kps[:, el, 1])


def _check_non_raised(kps, scores, cfg, side, pct):
    sh = cfg.left_shoulder if side == "LEFT" else cfg.right_shoulder
    wr = cfg.left_wrist if side == "LEFT" else cfg.right_wrist
    below = kps[:, wr, 1] >= kps[:, sh, 1]
    return len(below) == 0 or np.mean(below) >= pct


def _find_all_runs(mask, min_len):
    if len(mask) == 0:
        return []
    runs, start, length = [], 0, 0
    for i, v in enumerate(mask):
        if v:
            if length == 0:
                start = i
            length += 1
        else:
            if length >= min_len:
                runs.append((start, start + length))
            length = 0
    if length >= min_len:
        runs.append((start, start + length))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs


def find_plateau(wrist_y, cfg=None):
    """Find velocity plateau within a wrist_y signal."""
    cfg = cfg or HandFinderConfig()
    n = len(wrist_y)
    if n < 4:
        return None
    win = min(cfg.plateau_savgol_window, n)
    if win % 2 == 0:
        win -= 1
    poly = min(cfg.plateau_savgol_poly, win - 1)
    smoothed = savgol_filter(wrist_y, win, poly)
    velocity = np.abs(np.diff(smoothed))
    threshold = np.percentile(velocity, cfg.plateau_vel_percentile)
    run = _find_all_runs(velocity <= threshold, cfg.plateau_min_frames)
    return run[0] if run else None


def _pick_representative(kps, scores, seg_start, seg_end, side, cfg):
    indices = [cfg.left_shoulder, cfg.left_elbow, cfg.left_wrist] if side == "LEFT" \
        else [cfg.right_shoulder, cfg.right_elbow, cfg.right_wrist]
    wr_idx = cfg.left_wrist if side == "LEFT" else cfg.right_wrist

    search_start, search_end = seg_start, seg_end
    plateau = find_plateau(kps[seg_start:seg_end, wr_idx, 1], cfg)
    if plateau is not None:
        search_start = seg_start + plateau[0]
        search_end = seg_start + plateau[1]

    best_frame, best_score = search_start, -1.0
    for i in range(search_start, search_end):
        mc = min(scores[i, idx] for idx in indices)
        if mc > best_score:
            best_score = mc
            best_frame = i
    return best_frame


def _compute_crop_box(pkl_data, frame_idx, wrist_idx, cfg, meta):
    kp = pkl_data[f"frame_{frame_idx}"]["keypoints"]
    wx, wy = int(kp[wrist_idx][0]), int(kp[wrist_idx][1])
    pad = cfg.crop_padding
    return (
        max(0, wx - pad), max(0, wy - pad),
        min(meta.get("width", 1920), wx + pad),
        min(meta.get("height", 1080), wy + pad),
    )


def find_score_hands(contact_result, config=None):
    """Detect post-swing hand raises from keypoint data."""
    cfg = config or HandFinderConfig()
    ct = contact_result
    br = ct.backswing_result
    meta = br.pkl_data.get("__meta__", {})
    windows = _get_search_windows(ct)

    hand_frames, rep_frames, sides, crops, classes = [], [], [], [], []

    for window in windows:
        if window is None or window[1] <= window[0]:
            hand_frames.append(None); rep_frames.append(None)
            sides.append(None); crops.append(None)
            classes.append("practice_swing")
            continue

        start, end = window
        kps, scores = _extract_keypoints_window(br.pkl_data, start, end)

        left_runs = _find_all_runs(_check_raised(kps, scores, cfg, "LEFT"), cfg.min_raise_frames)
        right_runs = _find_all_runs(_check_raised(kps, scores, cfg, "RIGHT"), cfg.min_raise_frames)

        def best_run(runs, opp_side):
            for r in runs:
                if _check_non_raised(kps[r[0]:r[1]], scores[r[0]:r[1]], cfg, opp_side, cfg.non_raised_pct):
                    return r
            return None

        lr, rr = best_run(left_runs, "RIGHT"), best_run(right_runs, "LEFT")

        if lr is None and rr is None:
            hand_frames.append(None); rep_frames.append(None)
            sides.append(None); crops.append(None)
            classes.append("practice_swing")
            continue

        if lr and rr:
            side = "LEFT" if (lr[1] - lr[0]) >= (rr[1] - rr[0]) else "RIGHT"
            run = lr if side == "LEFT" else rr
        elif lr:
            side, run = "LEFT", lr
        else:
            side, run = "RIGHT", rr

        rep = _pick_representative(kps, scores, run[0], run[1], side, cfg)
        wr_idx = cfg.left_wrist if side == "LEFT" else cfg.right_wrist

        hand_frames.append((start + run[0], start + run[1] - 1))
        rep_frames.append(start + rep)
        sides.append(side)
        crops.append(_compute_crop_box(br.pkl_data, start + rep, wr_idx, cfg, meta))
        classes.append("scored_swing")

    return HandFinderResult(
        name=ct.name, hand_frames=hand_frames,
        representative_frames=rep_frames, raised_side=sides,
        crop_boxes=crops, classifications=classes, contact_result=ct,
    )
