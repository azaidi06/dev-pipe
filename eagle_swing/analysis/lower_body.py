"""Lower body metric computation.

Ported from eagle_swing/lower_body.py (computation only; plotting in viz/).
"""

import numpy as np

from ..utils import angle_3pts


def _lead_trail_lower(handedness="right"):
    if handedness.lower() == "right":
        return dict(lead_hip=11, trail_hip=12, lead_knee=13, trail_knee=14,
                    lead_ankle=15, trail_ankle=16)
    return dict(lead_hip=12, trail_hip=11, lead_knee=14, trail_knee=13,
                lead_ankle=16, trail_ankle=15)


def _fill_nan(pts, conf=None, thresh=0.3):
    filled = pts.copy()
    if conf is not None:
        filled[conf < thresh] = np.nan
    for i in range(1, len(filled)):
        if np.any(np.isnan(filled[i])):
            filled[i] = filled[i - 1]
    for i in range(len(filled) - 2, -1, -1):
        if np.any(np.isnan(filled[i])):
            filled[i] = filled[i + 1]
    return filled


def _mavg(pts, window=5):
    if window < 2:
        return pts
    smoothed = pts.copy()
    pad = window // 2
    for i in range(len(pts)):
        s, e = max(0, i - pad), min(len(pts), i + pad + 1)
        smoothed[i] = np.nanmean(pts[s:e], axis=0)
    return smoothed


def compute_lower_metrics(kps, scores=None, handedness="right",
                          min_conf=0.3, smooth_win=5):
    """Compute lower-body time-series metrics for one sequence.

    Returns dict with keys: hip_rotation_deg, lead_hip_lateral_shift,
    lead_knee_flexion, trail_knee_flexion, knee_hip_ratio,
    lead_ankle_sway, trail_ankle_sway.
    """
    T = kps.shape[0]
    smooth_lb = max(smooth_win * 2, 7)

    def prep(idx):
        v = kps[:, idx, :2].copy()
        c = scores[:, idx] if scores is not None else None
        return _mavg(_fill_nan(v, c, min_conf), smooth_lb)

    l_hip, r_hip = prep(11), prep(12)
    l_knee, r_knee = prep(13), prep(14)
    l_ankle, r_ankle = prep(15), prep(16)

    delta = l_hip - r_hip
    hip_rot = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))

    idxs = _lead_trail_lower(handedness)
    lead_hip = l_hip if handedness == "right" else r_hip
    addr_x = np.nanmedian(lead_hip[:min(10, T), 0])

    is_right = handedness == "right"
    lead_knee_flex = angle_3pts(
        l_hip if is_right else r_hip,
        l_knee if is_right else r_knee,
        l_ankle if is_right else r_ankle,
    )
    trail_knee_flex = angle_3pts(
        r_hip if is_right else l_hip,
        r_knee if is_right else l_knee,
        r_ankle if is_right else l_ankle,
    )

    knee_w = np.linalg.norm(l_knee - r_knee, axis=1)
    hip_w = np.linalg.norm(l_hip - r_hip, axis=1)

    lead_ankle = l_ankle if is_right else r_ankle
    trail_ankle = r_ankle if is_right else l_ankle

    return {
        "hip_rotation_deg": hip_rot,
        "lead_hip_lateral_shift": lead_hip[:, 0] - addr_x,
        "lead_knee_flexion": lead_knee_flex,
        "trail_knee_flexion": trail_knee_flex,
        "knee_hip_ratio": knee_w / (hip_w + 1e-6),
        "lead_ankle_sway": lead_ankle[:, 0] - np.nanmedian(lead_ankle[:min(10, T), 0]),
        "trail_ankle_sway": trail_ankle[:, 0] - np.nanmedian(trail_ankle[:min(10, T), 0]),
    }
