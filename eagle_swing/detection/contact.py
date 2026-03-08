"""Contact (impact) frame detection.

Public API:
    detect_contacts(backswing_result, config=None) -> ContactResult
"""

import numpy as np
from scipy.signal import savgol_filter

from .config import Config, ContactResult, DetectionResult


def _detect_contact_points(bs_frames, combined, n_pkl, cfg):
    """Returns array parallel to bs_frames; -1 where contact could not be found."""
    smoothed = savgol_filter(combined, cfg.contact_savgol_window, cfg.contact_savgol_poly)
    contacts = []
    for bf in bs_frames:
        s = bf + cfg.contact_search_min
        e = min(bf + cfg.contact_search_max, n_pkl - 1)
        if s >= n_pkl:
            contacts.append(-1)
            continue
        seg = smoothed[s:e + 1]
        if len(seg) == 0:
            contacts.append(-1)
            continue
        best = s + int(np.argmax(seg))
        # Refine: exact peak on unsmoothed combined signal within ±contact_refine_window
        rw = cfg.contact_refine_window
        ref_s = max(s, best - rw)
        ref_e = min(e + 1, best + rw + 1)
        ref_seg = combined[ref_s:ref_e]
        contacts.append(ref_s + int(np.argmax(ref_seg)))
    return np.array(contacts, dtype=int), smoothed


def _n_pkl_frames(pkl_data):
    return sum(1 for k in pkl_data if k.startswith("frame_"))


def detect_contacts(backswing_result: DetectionResult, config=None) -> ContactResult:
    """Detect contact/impact frames following each backswing.

    Parameters
    ----------
    backswing_result : DetectionResult
        Output from detect_backswings.
    config : Config, optional
        Detection config (uses defaults if None).

    Returns
    -------
    ContactResult
    """
    cfg = config or Config()
    br = backswing_result
    cf, sm = _detect_contact_points(
        br.peak_frames, br.combined, _n_pkl_frames(br.pkl_data), cfg
    )
    clog = []
    valid = cf[cf >= 0]
    n_dup = len(valid) - len(set(valid.tolist()))
    if n_dup > 0:
        clog.append(f"Contact: {n_dup} duplicate contact frame(s)")
    n_miss = int(np.sum(cf < 0))
    if n_miss > 0:
        clog.append(f"Contact: {n_miss} backswing(s) had no contact (near end of video)")
    return ContactResult(
        name=br.name, contact_frames=cf, backswing_result=br,
        smoothed=sm, filter_log=clog,
    )
