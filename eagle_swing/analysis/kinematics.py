"""Kinematic event detection: body-line crossing, top-of-backswing.

Ported from eagle_swing/temporal.py.
"""

import numpy as np
from scipy.signal import savgol_filter


def get_signed_distance(p_hand, p_shoulder, p_hip):
    """Signed distance of hand from the shoulder-hip body line.

    > 0 if hand is to the right of the line, < 0 if left.
    Uses vertical boundary when hand is above shoulder.
    """
    x_h, y_h = p_hand[:2]
    x_s, y_s = p_shoulder[:2]
    x_hip, y_hip = p_hip[:2]

    if y_h < y_s:
        return x_h - x_s
    if abs(y_hip - y_s) < 1e-6:
        return x_h - x_s

    slope_inv = (x_hip - x_s) / (y_hip - y_s)
    x_line = x_s + (y_h - y_s) * slope_inv
    return x_h - x_line


def find_crossing_frames(keypoints):
    """Find frames where hands cross body lines (right -> left direction).

    Returns dict with 'right_hand_crosses_left_body' and
    'left_hand_crosses_right_body' frame lists.
    """
    events = {"right_hand_crosses_left_body": [], "left_hand_crosses_right_body": []}
    holder = []

    for t in range(len(keypoints) - 1):
        curr, nxt = keypoints[t], keypoints[t + 1]

        # Right hand crossing left body line
        d_curr = get_signed_distance(curr[10], curr[5], curr[11])
        d_next = get_signed_distance(nxt[10], nxt[5], nxt[11])
        if d_curr > 0 and d_next <= 0:
            events["right_hand_crosses_left_body"].append(t + 1)
            holder.append(t)

        # Left hand crossing right body line
        d_curr_l = get_signed_distance(curr[9], curr[6], curr[12])
        d_next_l = get_signed_distance(nxt[9], nxt[6], nxt[12])
        if d_curr_l > 0 and d_next_l <= 0:
            events["left_hand_crosses_right_body"].append(t + 1)

    if len(holder) >= 2:
        events["time_between_crossing"] = holder[1] - holder[0]
    return events


def find_top_of_backswing(keypoints):
    """Find frame where hands transition from rising to falling (top of backswing).

    Returns frame index or None.
    """
    raw_ys = (keypoints[:, 9, 1] + keypoints[:, 10, 1]) / 2
    try:
        ys = savgol_filter(raw_ys, window_length=7, polyorder=2)
    except ValueError:
        ys = np.convolve(raw_ys, np.ones(5) / 5, mode="same")

    velocity = np.diff(ys)
    margin = len(velocity) // 10 if len(velocity) >= 20 else 0
    valid = velocity[margin:len(velocity) - margin] if margin else velocity
    if len(valid) == 0:
        return None

    max_ds = np.argmax(valid) + margin
    top = max_ds
    for t in range(max_ds, 0, -1):
        if velocity[t] <= 0.5:
            top = t
            break
    return top
