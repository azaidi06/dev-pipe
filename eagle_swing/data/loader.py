"""Unified pkl loader for both eagle-swing and pipeline formats.

Eagle-swing format: ordered dict, keys are arbitrary strings, each value
    has 'keypoints' (17, 2) and 'keypoint_scores' (17,). No '__meta__'.

Pipeline format: dict with 'frame_N' keys (N = 0, 1, ...), each value
    has 'keypoints' (17, 2) and 'keypoint_scores' (17,). Has '__meta__'
    dict with fps, total_frames, n_pkl_frames.
"""

import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
from scipy.signal import savgol_filter
import pandas as pd

from .schema import SwingMeta, KeypointData

_DEFAULT_DATA_ROOT = "/home/azaidi/Desktop/golf/data/full_videos/ymirza"
DATA_ROOT = os.environ.get("EAGLE_SWING_DATA", _DEFAULT_DATA_ROOT)


def load_pkl(pkl_path: Union[str, Path]) -> KeypointData:
    """Load a pkl file in either eagle-swing or pipeline format."""
    pkl_path = str(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    meta_dict = data.pop("__meta__", {}) if isinstance(data, dict) else {}

    if isinstance(data, dict):
        frame_keys = sorted(
            [k for k in data if k != "__meta__"],
            key=_frame_sort_key,
        )
    elif isinstance(data, list):
        frame_keys = list(range(len(data)))
    else:
        raise ValueError(f"Unsupported pkl structure: {type(data)}")

    kps = np.stack([data[k]["keypoints"] for k in frame_keys])
    scs = np.stack([data[k]["keypoint_scores"] for k in frame_keys])

    video_path = pkl_path.rsplit(".", 1)[0] + ".mp4"
    meta = SwingMeta(
        pkl_path=pkl_path,
        fps=meta_dict.get("fps"),
        total_frames=meta_dict.get("total_frames"),
        n_pkl_frames=len(frame_keys),
        video_path=video_path,
        extra=meta_dict,
    )
    return KeypointData(keypoints=kps, scores=scs, meta=meta)


def load_pkl_raw(pkl_path: Union[str, Path]) -> dict:
    """Load raw pkl dict (for pipeline functions that need the raw dict)."""
    with open(str(pkl_path), "rb") as f:
        return pickle.load(f)


def interpolate_and_smooth(kp_data: KeypointData, window: int = 7, poly: int = 3) -> KeypointData:
    """Interpolate missing keypoints and apply Savitzky-Golay smoothing."""
    raw = kp_data.raw.astype(float)
    n_frames, n_kps, _ = raw.shape
    if n_frames < window:
        return kp_data

    xy = raw[:, :, :2].reshape(n_frames, -1)
    df = pd.DataFrame(xy).interpolate(method="linear", limit_direction="both")
    smoothed = savgol_filter(df.to_numpy(), window_length=window, polyorder=poly, axis=0)
    new_kps = smoothed.reshape(n_frames, n_kps, 2)

    return KeypointData(
        keypoints=new_kps,
        scores=kp_data.scores.copy(),
        meta=kp_data.meta,
    )


def discover_sessions(root: str = DATA_ROOT):
    """Find session directories under the data root."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def _frame_sort_key(key):
    """Sort frame keys: 'frame_0', 'frame_1', ... or alphabetically."""
    if isinstance(key, str) and key.startswith("frame_"):
        try:
            return int(key.split("_", 1)[1])
        except ValueError:
            pass
    return key
