"""Shared utilities: COCO keypoint map and angle computations."""

import numpy as np

COCO_KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_sh": 5, "r_sh": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

COCO_BODY = {k: v for k, v in COCO_KP.items() if v >= 5}


def angle_2pts(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Angle (degrees) of vector p1->p2 relative to +x axis. Vectorized over frames."""
    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    angles = np.arctan2(dy, dx)
    np.degrees(angles, out=angles)
    return angles


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (degrees) at vertex b formed by a-b-c. Vectorized over frames."""
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)
    ba_norm[ba_norm == 0] = 1e-6
    bc_norm[bc_norm == 0] = 1e-6
    cos_angle = np.clip(np.sum(ba * bc, axis=1) / (ba_norm * bc_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))
