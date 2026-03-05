"""Swing event detection: address frame, takeaway, phases.

Ported from eagle_swing/swing_events.py. These functions operate on raw
keypoint arrays (N, 17, 3) or (N, 17, 2).
"""

import numpy as np
from scipy.signal import savgol_filter


def find_address_velocity(kps, fps=60, stillness_thresh=2.0, lookback_buffer=5):
    """Find address frame via wrist velocity drop-off.

    Walks backward from first significant movement to find stillness.
    """
    hands = (kps[:, 9, :2] + kps[:, 10, :2]) / 2.0
    velocity = np.sqrt(np.sum(np.diff(hands, axis=0) ** 2, axis=1))
    window = 5
    velocity_smooth = np.convolve(velocity, np.ones(window) / window, mode="same")

    movement_thresh = stillness_thresh * 3.0
    movement_start_idx = -1
    for i in range(len(velocity_smooth)):
        if velocity_smooth[i] > movement_thresh:
            if np.mean(velocity_smooth[i : i + 5]) > movement_thresh:
                movement_start_idx = i
                break

    if movement_start_idx == -1:
        return 0

    address_idx = movement_start_idx
    for i in range(movement_start_idx, 0, -1):
        if velocity_smooth[i] < stillness_thresh:
            address_idx = i
            break

    return max(0, address_idx - lookback_buffer)


def find_address_robust(kps, fps=60, lookahead_frames=10):
    """Scale-invariant address finder using body-height normalization."""
    valid_frames = min(30, len(kps))
    nose = kps[:valid_frames, 0, :2]
    l_ankle = kps[:valid_frames, 15, :2]
    r_ankle = kps[:valid_frames, 16, :2]

    if np.mean(l_ankle) < 1e-3:
        lower_body = (kps[:valid_frames, 11, :2] + kps[:valid_frames, 12, :2]) / 2.0
    else:
        lower_body = (l_ankle + r_ankle) / 2.0

    ref_height = np.mean(np.linalg.norm(nose - lower_body, axis=1))
    if ref_height == 0:
        ref_height = 1.0

    hands = (kps[:, 9, :2] + kps[:, 10, :2]) / 2.0
    window_len = max(7, int(fps * 0.1) | 1)
    hands_smooth = savgol_filter(hands, window_length=window_len, polyorder=3, axis=0)

    raw_disp = np.linalg.norm(
        np.diff(hands_smooth, axis=0, prepend=hands_smooth[0:1]), axis=1
    )
    velocity_norm = raw_disp / ref_height

    noise_floor = np.percentile(velocity_norm, 10)
    trigger_thresh = max(noise_floor * 3.0, 0.005)

    candidates = np.where(velocity_norm > trigger_thresh)[0]
    true_takeaway_idx = -1

    for idx in candidates:
        if idx < 10:
            continue
        if idx + lookahead_frames >= len(hands_smooth):
            continue
        start_pt = hands_smooth[idx]
        end_pt = hands_smooth[idx + lookahead_frames]
        net_disp = np.linalg.norm(end_pt - start_pt) / ref_height
        segment = hands_smooth[idx : idx + lookahead_frames + 1]
        path_len = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)) / ref_height
        if path_len < 1e-5:
            continue
        ratio = net_disp / path_len
        if ratio > 0.80 and net_disp > 0.02:
            true_takeaway_idx = idx
            break

    if true_takeaway_idx == -1:
        return 0

    back_buffer = int(fps * 0.5)
    search_start = max(0, true_takeaway_idx - back_buffer)
    local_min = np.argmin(velocity_norm[search_start : true_takeaway_idx + 1])
    return search_start + local_min


def find_address_optimized(kps, fps=60, lookback_seconds=1.2):
    """Find address via minimum velocity before the main swing event."""
    hands = (kps[:, 9, :2] + kps[:, 10, :2]) / 2.0
    deltas = np.diff(hands, axis=0, prepend=hands[0:1])
    speed = np.sqrt(np.sum(deltas ** 2, axis=1))

    window_len = max(5, int(fps * 0.05) | 1)
    speed_smooth = savgol_filter(speed, window_len, 2) if len(speed) > window_len else speed

    peak_velocity = np.percentile(speed_smooth, 98)
    trigger_thresh = peak_velocity * 0.40
    candidates = np.where(speed_smooth > trigger_thresh)[0]

    if len(candidates) == 0:
        return 0

    trigger_idx = candidates[0]
    lookback_frames = int(fps * lookback_seconds)
    search_start = max(0, trigger_idx - lookback_frames)

    window_speed = speed_smooth[search_start : trigger_idx + 1]
    min_local_idx = np.argmin(window_speed)
    address_frame = search_start + min_local_idx

    if speed_smooth[address_frame] > (peak_velocity * 0.1):
        return 0

    return address_frame
