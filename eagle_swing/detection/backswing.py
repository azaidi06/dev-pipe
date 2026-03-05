"""Backswing apex detection via wrist signal processing.

Public API:
    detect_backswings(pkl_path, mov_path=None, *, fps=None, total_frames=None, config=None)
        -> DetectionResult
"""

import os
import pickle

import numpy as np
from scipy.signal import savgol_filter, find_peaks

from .config import Config, DetectionResult


def _load_wrist_signals(pkl_path, cfg):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    n = sum(1 for k in data if k.startswith("frame_"))
    kps = np.array([data[f"frame_{i}"]["keypoints"] for i in range(n)])
    scs = np.array([data[f"frame_{i}"]["keypoint_scores"] for i in range(n)])
    return (
        kps[:, cfg.left_wrist, 0], kps[:, cfg.right_wrist, 0],
        kps[:, cfg.left_wrist, 1], kps[:, cfg.right_wrist, 1],
        scs[:, cfg.left_wrist], scs[:, cfg.right_wrist], data,
    )


def _interp_low_conf(signal, conf, threshold):
    bad = conf < threshold
    if not np.any(bad):
        return signal.copy()
    good = ~bad
    if np.sum(good) < 2:
        return signal.copy()
    out = signal.copy()
    out[np.where(bad)[0]] = np.interp(
        np.where(bad)[0], np.where(good)[0], signal[good]
    )
    return out


def _build_combined_signal(pkl_path, cfg):
    x_l, x_r, y_l, y_r, c_l, c_r, data = _load_wrist_signals(pkl_path, cfg)
    x_l = _interp_low_conf(x_l, c_l, cfg.conf_threshold)
    x_r = _interp_low_conf(x_r, c_r, cfg.conf_threshold)
    y_l = _interp_low_conf(y_l, c_l, cfg.conf_threshold)
    y_r = _interp_low_conf(y_r, c_r, cfg.conf_threshold)
    return (x_l + x_r) / 2.0 + (y_l + y_r) / 2.0, data


def _backswing_score(peak, smoothed, cfg):
    behind = smoothed[max(0, peak - cfg.look_behind):peak]
    approach_jitter = np.std(np.diff(behind)) if len(behind) > 2 else 0.0
    ahead = smoothed[peak:min(len(smoothed), peak + cfg.look_ahead)]
    departure_drop = (ahead[-1] - ahead[0]) if len(ahead) > 1 else 0.0
    return approach_jitter - 0.5 * departure_drop


def _detect_peaks(combined, total_video_frames, cfg):
    smoothed = savgol_filter(combined, cfg.savgol_window, cfg.savgol_poly)
    coarse = savgol_filter(combined, cfg.coarse_window, cfg.coarse_poly)
    neg = -smoothed
    if total_video_frames is not None:
        ms = int(total_video_frames * (1.0 - cfg.end_of_video_pct))
        if ms < len(neg):
            neg[ms:] = np.min(neg)
    anchors, _ = find_peaks(neg, prominence=cfg.peak_prominence, distance=cfg.peak_distance)
    if len(anchors) == 0:
        return anchors, smoothed
    results = []
    for anchor in anchors:
        ss = max(0, anchor - cfg.search_back)
        d = np.diff(coarse[ss:anchor + 1])
        candidates = [ss + i + 1 for i in range(len(d) - 1) if d[i] <= 0 and d[i + 1] > 0]
        if anchor not in candidates:
            candidates.append(anchor)
        best = candidates[np.argmin([_backswing_score(c, smoothed, cfg) for c in candidates])]
        refine = smoothed[best:min(len(smoothed), best + cfg.refine_window + 1)]
        results.append(best + int(np.argmin(refine)))
    return np.array(results), smoothed


def _filter_peaks(peaks, smoothed, total_frames, pkl_data, cfg):
    log = []
    before = len(peaks)
    peaks = np.array(sorted(set(peaks.tolist())))
    if len(peaks) != before:
        log.append(f"Dedup: removed {before - len(peaks)} duplicate(s)")
    if len(peaks) == 0:
        return peaks, log
    # End-of-video trim
    cutoff = int(total_frames * (1.0 - cfg.end_of_video_pct))
    mask = peaks < cutoff
    if not np.all(mask):
        log.append(f"End-of-video: removed {np.sum(~mask)} peak(s) past frame {cutoff}")
        peaks = peaks[mask]
    if len(peaks) == 0:
        return peaks, log
    # Too-close merge
    merged, removed_close = [peaks[0]], []
    for p in peaks[1:]:
        if p - merged[-1] < cfg.min_swing_gap:
            prev = merged[-1]
            if smoothed[p] < smoothed[prev]:
                removed_close.append(prev)
                merged[-1] = p
            else:
                removed_close.append(p)
        else:
            merged.append(p)
    if removed_close:
        log.append(f"Too-close: removed {len(removed_close)} peak(s) within {cfg.min_swing_gap} frames")
    peaks = np.array(merged)
    if len(peaks) < cfg.xy_outlier_min_peaks:
        return peaks, log
    # x+y outlier removal (MAD)
    vals = smoothed[peaks]
    med = np.median(vals)
    mad = max(np.median(np.abs(vals - med)), cfg.xy_outlier_mad_floor)
    if mad > 0:
        thresh = med + cfg.xy_outlier_mad_thresh * mad
        om = vals > thresh
        if np.any(om):
            log.append(f"MAD: removed {np.sum(om)} peak(s) with x+y > {thresh:.0f} (med={med:.0f}, MAD={mad:.0f})")
            peaks = peaks[~om]
    # Follow-through rejection
    if pkl_data is not None and len(peaks) >= cfg.xy_outlier_min_peaks:
        offsets = np.empty(len(peaks))
        for i, p in enumerate(peaks):
            kp = np.array(pkl_data[f"frame_{p}"]["keypoints"])
            offsets[i] = (
                (kp[cfg.left_wrist][0] + kp[cfg.right_wrist][0]) / 2
                - (kp[cfg.left_shoulder][0] + kp[cfg.right_shoulder][0]) / 2
            )
        med_off = np.median(offsets)
        mad_off = max(np.median(np.abs(offsets - med_off)), cfg.xoff_mad_floor)
        thresh_off = med_off + cfg.xoff_mad_thresh * mad_off
        om = offsets > thresh_off
        if np.any(om):
            log.append(f"Follow-through: removed {np.sum(om)} peak(s) with x_offset > {thresh_off:.0f}")
            peaks = peaks[~om]
    return peaks, log


def detect_backswings(pkl_path, mov_path=None, *, fps=None, total_frames=None, config=None):
    """Detect backswing apex frames from a pkl keypoint file.

    Parameters
    ----------
    pkl_path : str
        Path to pkl file (pipeline format with frame_N keys).
    mov_path : str, optional
        Path to video file (used to extract fps/total_frames if not in pkl __meta__).
    fps : float, optional
        Frames per second (overrides pkl metadata).
    total_frames : int, optional
        Total video frames (overrides pkl metadata).
    config : Config, optional
        Detection config (uses defaults if None).

    Returns
    -------
    DetectionResult
    """
    cfg = config or Config()
    combined, pkl_data = _build_combined_signal(pkl_path, cfg)

    meta = pkl_data.get("__meta__", {})
    fps = fps or meta.get("fps")
    total_frames = total_frames or meta.get("total_frames")

    if fps is None or total_frames is None:
        if mov_path is None:
            raise ValueError(
                "fps and total_frames must be provided (directly or via pkl __meta__) "
                "when mov_path is not given"
            )
        import cv2
        cap = cv2.VideoCapture(mov_path)
        fps = fps or cap.get(cv2.CAP_PROP_FPS)
        total_frames = total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    total_frames = int(total_frames)
    name = os.path.splitext(os.path.basename(mov_path or pkl_path))[0]

    peak_frames, smoothed = _detect_peaks(combined, total_frames, cfg)
    peak_frames, flog = _filter_peaks(peak_frames, smoothed, total_frames, pkl_data, cfg)
    return DetectionResult(
        name=name,
        peak_frames=peak_frames, smoothed=smoothed, combined=combined,
        fps=fps, total_frames=total_frames, filter_log=flog,
        pkl_data=pkl_data, pkl_path=pkl_path, mov_path=mov_path or "",
    )
