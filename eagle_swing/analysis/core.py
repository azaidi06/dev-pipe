"""Core swing analysis: metrics, SPM, SwingData, resampling.

Ported from golf-pipeline-final/analyze/core.py with deduplication
against eagle_swing body modules.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, resample
from scipy.stats import ttest_ind, f as f_dist

from ..utils import COCO_KP, COCO_BODY, angle_3pts, angle_2pts

# Signal processing defaults
SAVGOL_WINDOW = 7
SAVGOL_POLY = 3
VELOCITY_WINDOW = 5
SPEED_SMOOTHING_WINDOW = 11
DEFAULT_FPS = 60
RESAMPLE_TARGET = 100
SPM_ALPHA = 0.05
MIN_SIG_DURATION_PCT = 4

METRIC_GROUPS = {
    "Rotation": [
        "Shoulder Turn (Proxy)", "Hip Turn (Proxy)", "X-Factor (Stretch)",
        "Shoulder Rot. Vel.", "Hip Rot. Vel.",
    ],
    "Posture": [
        "Spine Angle", "Shoulder Tilt", "Hip Tilt",
        "Lead Knee Angle", "Trail Knee Angle", "Right Elbow Angle",
    ],
    "Linear": [
        "Hip Sway (X)", "Head Sway (X)", "Head Height (Y)",
        "Hand Height (Y)", "Hand Depth (X)", "Hand-Body Dist",
    ],
}
ALL_METRICS = [m for g in METRIC_GROUPS.values() for m in g]


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def calculate_slope(a, b):
    """Slope angle (degrees) of vector a->b relative to horizontal."""
    return np.degrees(np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0]))


def get_velocity(data, window=VELOCITY_WINDOW):
    """1st derivative via Savgol filter."""
    if len(data) < window:
        return np.zeros_like(data)
    try:
        return savgol_filter(data, window, 2, deriv=1)
    except ValueError:
        return np.gradient(data, axis=0)


def get_kp_speeds(keypoints, window_size=SPEED_SMOOTHING_WINDOW, fps=DEFAULT_FPS):
    velocities = np.gradient(keypoints, axis=0) / (1.0 / fps)
    speeds = np.linalg.norm(velocities, axis=1)
    return np.convolve(speeds, np.ones(window_size) / window_size, mode="same")


def compute_swing_metrics(kps):
    """Compute biomechanical metrics from (N, 17, 2) keypoints."""
    hand_height = -kps[:, 10, 1]
    hand_depth = kps[:, 10, 0]

    l_hip, l_knee, l_ankle = kps[:, 11], kps[:, 13], kps[:, 15]
    lead_knee = angle_3pts(l_hip, l_knee, l_ankle)

    mid_hip = (kps[:, 11] + kps[:, 12]) / 2
    mid_sho = (kps[:, 5] + kps[:, 6]) / 2
    spine_vec = mid_sho - mid_hip
    vertical = np.tile([0, -1], (len(kps), 1)).astype(float)
    norm_spine = np.linalg.norm(spine_vec, axis=1)
    valid = norm_spine > 1e-5

    spine_angle = np.zeros(len(kps))
    if np.any(valid):
        cos_s = np.sum(spine_vec[valid] * vertical[valid], axis=1) / norm_spine[valid]
        spine_angle[valid] = np.degrees(np.arccos(np.clip(cos_s, -1.0, 1.0)))

    r_hip, r_knee, r_ankle = kps[:, 12], kps[:, 14], kps[:, 16]
    trail_knee = angle_3pts(r_hip, r_knee, r_ankle)

    r_sho, r_elbow, r_wrist = kps[:, 6], kps[:, 8], kps[:, 10]
    elbow = angle_3pts(r_sho, r_elbow, r_wrist)

    shoulder_tilt = calculate_slope(kps[:, 5], kps[:, 6])
    hip_tilt = calculate_slope(kps[:, 11], kps[:, 12])

    hand_body_dist = np.linalg.norm(kps[:, 10] - mid_hip, axis=1)

    sho_w = np.linalg.norm(kps[:, 5] - kps[:, 6], axis=1)
    hip_w = np.linalg.norm(kps[:, 11] - kps[:, 12], axis=1)
    sho_turn = sho_w[0] - sho_w
    hip_turn = hip_w[0] - hip_w
    x_factor = sho_turn - hip_turn

    return {
        "Hand Depth (X)": hand_depth,
        "Hand Height (Y)": hand_height,
        "Spine Angle": spine_angle,
        "Hip Sway (X)": mid_hip[:, 0],
        "Head Sway (X)": kps[:, 0, 0],
        "Head Height (Y)": -kps[:, 0, 1],
        "Lead Knee Angle": lead_knee,
        "Trail Knee Angle": trail_knee,
        "Right Elbow Angle": elbow,
        "Shoulder Tilt": shoulder_tilt,
        "Hip Tilt": hip_tilt,
        "Hand-Body Dist": hand_body_dist,
        "Shoulder Turn (Proxy)": sho_turn,
        "Hip Turn (Proxy)": hip_turn,
        "X-Factor (Stretch)": x_factor,
        "Shoulder Rot. Vel.": get_velocity(sho_turn),
        "Hip Rot. Vel.": get_velocity(hip_turn),
    }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def interpolate_and_filter(keypoints, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    """Interpolate NaNs and Savgol-smooth keypoints (N, K, 2+)."""
    n_frames, n_kps, n_coords = keypoints.shape
    xy = keypoints[:, :, :2].reshape(n_frames, -1)
    df = pd.DataFrame(xy).interpolate(method="linear", limit_direction="both")
    smoothed = savgol_filter(df.to_numpy(), window_length=window, polyorder=poly, axis=0)
    result = smoothed.reshape(n_frames, n_kps, 2)
    if n_coords > 2:
        return np.concatenate([result, keypoints[:, :, 2:]], axis=2)
    return result


def normalize_scale_center(kps, ref_frame_idx=0):
    """Scale by median torso height, center on ref-frame hip."""
    l_shldr, r_shldr = kps[:, 5, :2], kps[:, 6, :2]
    l_hip, r_hip = kps[:, 11, :2], kps[:, 12, :2]
    mid_shldr = (l_shldr + r_shldr) / 2
    mid_hip = (l_hip + r_hip) / 2

    torso_lens = np.linalg.norm(mid_shldr - mid_hip, axis=1)
    scale = np.median(torso_lens[torso_lens > 0])
    if scale < 1e-5:
        return kps

    ref_origin = mid_hip[ref_frame_idx].reshape(1, 1, 2)
    out = kps.copy()
    out[:, :, :2] = (out[:, :, :2] - ref_origin) / scale
    return out


# ---------------------------------------------------------------------------
# SPM
# ---------------------------------------------------------------------------

def perform_spm_1d(group_a, group_b, alpha=SPM_ALPHA):
    arr_a, arr_b = np.vstack(group_a), np.vstack(group_b)
    if not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        return 0, False, np.ones(arr_a.shape[1])
    t_stat, p_val = ttest_ind(arr_a, arr_b, axis=0, equal_var=False)
    return t_stat, p_val < alpha, p_val


def perform_hotellings_t2(group_a, group_b, alpha=SPM_ALPHA):
    A, B = np.stack(group_a), np.stack(group_b)
    n1, n_frames, p = A.shape
    n2 = B.shape[0]
    if n1 < 2 or n2 < 2:
        return np.zeros(n_frames, dtype=bool)

    mean_a, mean_b = np.mean(A, axis=0), np.mean(B, axis=0)
    sig = []
    for i in range(n_frames):
        da, db = A[:, i, :], B[:, i, :]
        sp = ((n1 - 1) * np.cov(da, rowvar=False) + (n2 - 1) * np.cov(db, rowvar=False)) / (n1 + n2 - 2)
        diff = mean_a[i] - mean_b[i]
        try:
            inv_sp = np.linalg.inv(sp)
            t2 = (n1 * n2) / (n1 + n2) * diff.T @ inv_sp @ diff
            f_stat = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * t2
            crit = f_dist.ppf(1 - alpha, p, n1 + n2 - p - 1)
            sig.append(f_stat > crit)
        except np.linalg.LinAlgError:
            sig.append(False)
    return np.array(sig)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_1d(arr, target=RESAMPLE_TARGET):
    return resample(arr, target)


def resample_xy(x, y, target):
    idx = np.linspace(0, 1, len(x))
    t_idx = np.linspace(0, 1, target)
    return np.interp(t_idx, idx, x), np.interp(t_idx, idx, y)


def filter_noise(mask, min_duration=MIN_SIG_DURATION_PCT):
    if not np.any(mask):
        return mask
    cleaned = mask.copy()
    padded = np.concatenate(([False], cleaned, [False]))
    diff = np.diff(padded.astype(int))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) < min_duration:
            cleaned[s:e] = False
    return cleaned


def get_significant_regions(mask, label="Metric"):
    regions = []
    if not np.any(mask):
        return regions
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        regions.append({"Metric": label, "Start (%)": s, "End (%)": e, "Duration (%)": e - s})
    return regions


def find_swing_medoid(swings):
    n = len(swings)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(swings[i] - swings[j])
            dmat[i, j] = dmat[j, i] = d
    return np.argmin(np.sum(dmat, axis=1))


# ---------------------------------------------------------------------------
# SwingData
# ---------------------------------------------------------------------------

class SwingData:
    """Single swing segment with computed metrics."""

    def __init__(self, row, **kwargs):
        self.metadata = row
        self.metadata_names = list(self.metadata.keys()) if hasattr(self.metadata, "keys") else []
        self.og_kps = self._get_keypoints()
        self.kps = normalize_scale_center(interpolate_and_filter(self.og_kps.copy()))
        self.metrics = compute_swing_metrics(self.kps[:, :, :2])
        self.backswing = self.kps[:self.metadata.backswing_frame_count].copy()
        self.downswing = self.kps[self.metadata.backswing_frame_count:].copy()

    def _get_keypoints(self):
        with open(self.metadata.pkl_path, "rb") as fh:
            data = pickle.load(fh)
        frame_keys = sorted(
            [k for k in data if isinstance(k, str) and k.startswith("frame_")],
            key=lambda k: int(k.split("_")[1]),
        )
        kps = np.stack([data[k]["keypoints"] for k in frame_keys])
        scores = np.stack([data[k]["keypoint_scores"] for k in frame_keys])
        full = np.concatenate([kps, scores[..., None]], axis=2)
        s = max(0, self.metadata.top_backswing_idx - self.metadata.backswing_frame_count)
        e = min(len(full), self.metadata.top_backswing_idx + self.metadata.downswing_frame_count)
        return full[s:e]


# ---------------------------------------------------------------------------
# Group analysis
# ---------------------------------------------------------------------------

def perform_group_analysis(src_swings, tgt_swings, keys, mode="1d",
                           phase="backswing", coco_map=None, enable_spm=True,
                           target_frames=RESAMPLE_TARGET):
    if coco_map is None:
        coco_map = COCO_BODY
    results = {}
    do_spm = enable_spm and len(src_swings) > 1 and len(tgt_swings) > 1

    def extract(swings, key):
        out = []
        for s in swings:
            raw = None
            if mode == "2d":
                phase_data = getattr(s, phase)
                if phase_data is not None and len(phase_data) > 0:
                    raw = phase_data[:, key, :2]
            elif mode == "1d":
                if isinstance(key, str) and key in s.metrics:
                    raw = s.metrics[key]
                elif isinstance(key, int):
                    phase_data = getattr(s, phase)
                    if phase_data is not None and len(phase_data) > 0:
                        raw = get_kp_speeds(phase_data[:, key, :])
            if raw is not None and len(raw) > 0:
                out.append(resample_1d(raw, target_frames))
        return out

    for k in keys:
        src_data, tgt_data = extract(src_swings, k), extract(tgt_swings, k)
        res = {"sig": np.zeros(target_frames, dtype=bool), "src_mean": None}
        if src_data and tgt_data:
            res["src_mean"] = np.mean(np.stack(src_data), axis=0)
            if do_spm:
                if mode == "2d":
                    res["sig"] = perform_hotellings_t2(src_data, tgt_data)
                else:
                    _, sig, p_val = perform_spm_1d(src_data, tgt_data)
                    res["sig"] = sig
                    res["p_val"] = p_val
        results[k] = res
    return results


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_golfer_data(csv_path, golfer, data_dir, back_frames=60, down_frames=30, day_filter=None):
    df = pd.read_csv(csv_path)
    if "swing_index" in df.columns and "swing_idx" not in df.columns:
        df = df.rename(columns={"swing_index": "swing_idx"})
    if "top_backswing_idx" not in df.columns:
        raise KeyError("'top_backswing_idx' column missing from CSV")

    if golfer == "ymirza":
        base = os.path.join(data_dir, "..", "full_videos", "ymirza")
        df["day"] = df.pkl_path.map(lambda x: x.split("/")[0])
        df["pkl_path"] = df.pkl_path.map(lambda x: os.path.join(base, x))
    else:
        df["day"] = None
        df["pkl_path"] = df.pkl_path.map(lambda x: os.path.join(data_dir, x.split("/")[-1]))

    df["backswing_frame_count"] = back_frames
    df["downswing_frame_count"] = down_frames
    df["overlap_frame_count"] = 0
    if day_filter:
        df = df[df["day"].isin(day_filter)]
    return df


def build_swing_items(df):
    return [(i, SwingData(r)) for i, r in df.iterrows()]


def resolve_group_by_score(swing_items, score):
    score = int(score)
    return [s for _, s in swing_items if int(s.metadata.get("score", -1)) == score]


def get_medoid_from_pool(pool):
    if not pool:
        return None
    try:
        idx = find_swing_medoid([s.kps[:, :, :2] for s in pool])
        return pool[idx] if 0 <= idx < len(pool) else None
    except (ValueError, IndexError):
        return None
