"""Keypoint normalization strategies.

Merged from eagle_swing/normalization.py and analyze/core.py.
All functions operate on (N, 17, 2) or (N, 17, 3) arrays.
"""

import numpy as np


def normalize_by_torso_diagonal(kps, l_sh_to_r_hip=True):
    """Scale by one torso diagonal (L_SH->R_HIP or R_SH->L_HIP)."""
    if l_sh_to_r_hip:
        shoulder, hip = kps[:, 5, :2], kps[:, 12, :2]
    else:
        shoulder, hip = kps[:, 6, :2], kps[:, 11, :2]

    diag = np.linalg.norm(shoulder - hip, axis=1)
    scale = diag[:, None, None] + 1e-6

    xy = kps[..., :2] / scale
    if kps.shape[-1] > 2:
        return np.concatenate([xy, kps[..., 2:]], axis=-1)
    return xy


def normalize_by_average_torso(kps):
    """Scale by mean of both torso diagonals (more rotation-robust)."""
    d1 = np.linalg.norm(kps[:, 5, :2] - kps[:, 12, :2], axis=1)
    d2 = np.linalg.norm(kps[:, 6, :2] - kps[:, 11, :2], axis=1)
    scale = ((d1 + d2) / 2.0)[:, None, None] + 1e-6

    xy = kps[..., :2] / scale
    if kps.shape[-1] > 2:
        return np.concatenate([xy, kps[..., 2:]], axis=-1)
    return xy


def center_by_average_torso(kps):
    """Center on mid-hip and scale by average torso diagonal."""
    d1 = np.linalg.norm(kps[:, 5, :2] - kps[:, 12, :2], axis=1)
    d2 = np.linalg.norm(kps[:, 6, :2] - kps[:, 11, :2], axis=1)
    scale = ((d1 + d2) / 2.0)[:, None, None] + 1e-6

    hip_center = ((kps[:, 11, :2] + kps[:, 12, :2]) / 2.0)[:, None, :]
    xy = (kps[..., :2] - hip_center) / scale
    if kps.shape[-1] > 2:
        return np.concatenate([xy, kps[..., 2:]], axis=-1)
    return xy


def normalize_scale_center_robust(kps, ref_frame_idx=0):
    """Hybrid: median torso height scale + frame-0 hip centering."""
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


def align_to_body_frame_static(kps):
    """Center on mid-hip and correct camera tilt using frame-0 shoulder line."""
    xy = kps[..., :2].copy()
    mid_hip = ((xy[:, 11] + xy[:, 12]) / 2.0)[:, None, :]
    xy -= mid_hip

    shoulder_vec = xy[0, 5] - xy[0, 6]
    angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    xy = xy @ R.T

    if kps.shape[-1] > 2:
        return np.concatenate([xy, kps[..., 2:]], axis=-1)
    return xy


def normalize_static_vertical(kps):
    """Rotate all frames so frame-0 gravity line is vertical."""
    setup = kps[0]
    mid_hip = (setup[11, :2] + setup[12, :2]) / 2.0
    mid_ankle = (setup[15, :2] + setup[16, :2]) / 2.0
    body_vec = mid_hip - mid_ankle
    current = np.arctan2(body_vec[1], body_vec[0])
    rot = -np.pi / 2 - current
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])

    xy = np.einsum("ij,tfj->tfi", R, kps[..., :2])
    if kps.shape[-1] > 2:
        return np.concatenate([xy, kps[..., 2:]], axis=-1)
    return xy


def rescale_for_visualization(norm_kps, canvas_size=512, zoom_factor=200):
    """Map normalized coords back to pixel space for display."""
    vis = norm_kps.copy()
    vis[..., :2] = vis[..., :2] * zoom_factor + canvas_size / 2
    return vis
