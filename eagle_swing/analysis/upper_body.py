"""Upper body metric computation.

Ported from eagle_swing/upper_body.py (computation only; plotting in viz/).
"""

import numpy as np


def _lead_trail(handedness="right"):
    if handedness.lower().startswith("r"):
        return dict(lead_sh=5, lead_el=7, lead_wr=9, trail_sh=6, trail_el=8, trail_wr=10)
    return dict(lead_sh=6, lead_el=8, lead_wr=10, trail_sh=5, trail_el=7, trail_wr=9)


def _fill_low_conf(vals, conf=None, min_conf=0.35):
    vals = vals.astype(float, copy=True)
    T, D = vals.shape
    if conf is not None:
        bad = (conf.ravel()[:T] < min_conf)[:, None].repeat(D, axis=1)
        bad |= ~np.isfinite(vals)
    else:
        bad = ~np.isfinite(vals)
    for t in range(1, T):
        vals[t][bad[t]] = vals[t - 1][bad[t]]
    for t in range(T - 2, -1, -1):
        vals[t][bad[t]] = vals[t + 1][bad[t]]
    return vals


def _mavg(x, win=9):
    win = max(1, int(win) | 1)
    if x.ndim == 1:
        pad = win // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        return np.convolve(xp, np.ones(win) / win, mode="valid")
    out = np.empty_like(x, dtype=float)
    for d in range(x.shape[1]):
        out[:, d] = _mavg(x[:, d], win)
    return out


def compute_upper_metrics(kps, scores=None, handedness="right",
                          min_conf=0.35, smooth_win=9, invert_y=True):
    """Compute upper-body time-series metrics for one sequence.

    Returns dict with keys: wrist_y_lead, wrist_y_trail,
    shoulder_angle_deg, shoulder_delta_deg, wrist_x_separation, wrist_y_diff.
    """
    idxs = _lead_trail(handedness)
    T = kps.shape[0]

    def prep(idx):
        v = kps[:, idx, :2].copy()
        c = scores[:, idx] if scores is not None else None
        v = _fill_low_conf(v, c, min_conf)
        return _mavg(v, smooth_win)

    shL, shT = prep(idxs["lead_sh"]), prep(idxs["trail_sh"])
    wrL, wrT = prep(idxs["lead_wr"]), prep(idxs["trail_wr"])

    sgn = -1.0 if invert_y else 1.0
    dxy = shL - shT
    shoulder_deg = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    base = np.median(shoulder_deg[:min(T, 5)])

    return {
        "wrist_y_lead": sgn * wrL[:, 1],
        "wrist_y_trail": sgn * wrT[:, 1],
        "shoulder_angle_deg": shoulder_deg,
        "shoulder_delta_deg": shoulder_deg - base,
        "wrist_x_separation": np.abs(wrL[:, 0] - wrT[:, 0]),
        "wrist_y_diff": sgn * wrL[:, 1] - sgn * wrT[:, 1],
    }
