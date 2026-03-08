"""2D ensemble refinement for backswing apex and contact detection.

Ported from eagle-swing/posts/refine_comparison research notebook.
Provides alternative refinement that uses 2D wrist trajectory geometry
instead of 1D combined signal argmin.

Public API:
    refine_backswing_apex(wrist_window, method='ensemble', voting='mean')
    refine_contact_point(wrist_window, method='ensemble', voting='mean')
    get_all_method_picks(wrist_window, direction='backswing')
"""

import numpy as np
from scipy.signal import savgol_filter


# --- Individual 2D methods for backswing apex ---

def _argmin_y(wrist_xy):
    """Highest vertical point (min y in image coords)."""
    return int(np.argmin(wrist_xy[:, 1]))


def _argmin_xy_sum(wrist_xy):
    """Top-left corner of arc: argmin(x + y)."""
    return int(np.argmin(wrist_xy[:, 0] + wrist_xy[:, 1]))


def _velocity_zero_crossing(wrist_xy, thresh_frac=0.001):
    """Frame where vertical velocity reverses (upward -> downward).

    Returns None if no clear zero-crossing found.
    """
    if len(wrist_xy) < 12:
        return None
    pos = wrist_xy[:, :2].astype(float)
    wl = min(9, len(pos) - 1)
    if wl < 5 or wl % 2 == 0:
        wl = max(5, wl if wl % 2 == 1 else wl - 1)
    if len(pos) <= wl:
        return None
    pos_s = savgol_filter(pos, window_length=wl, polyorder=min(2, wl - 1), axis=0)
    v = np.diff(pos_s, axis=0)
    speed = np.linalg.norm(v, axis=1)
    max_speed = speed.max()
    if max_speed == 0:
        return None
    thresh = thresh_frac * max_speed
    vy = v[:, 1]
    sign = np.sign(vy)
    cross = np.where(
        (sign[:-1] < 0) & (sign[1:] > 0) &
        (speed[:-1] > thresh) & (speed[1:] > thresh)
    )[0]
    return int(cross[0] + 1) if len(cross) > 0 else None


# --- Individual 2D methods for contact point ---

def _argmax_y(wrist_xy):
    """Lowest vertical point (max y in image coords)."""
    return int(np.argmax(wrist_xy[:, 1]))


def _argmax_xy_sum(wrist_xy):
    """Bottom-right corner of arc: argmax(x + y)."""
    return int(np.argmax(wrist_xy[:, 0] + wrist_xy[:, 1]))


def _argmax_x(wrist_xy):
    """Furthest forward (max x)."""
    return int(np.argmax(wrist_xy[:, 0]))


# --- Voting strategies ---

def _vote_mean(picks):
    return int(round(np.mean(picks)))


def _vote_median(picks):
    return int(round(np.median(picks)))


def _vote_majority(picks):
    """Mode / majority vote. Ties broken by mean."""
    vals, counts = np.unique(picks, return_counts=True)
    max_count = counts.max()
    winners = vals[counts == max_count]
    return int(round(np.mean(winners)))


VOTING_STRATEGIES = {
    'mean': _vote_mean,
    'median': _vote_median,
    'majority': _vote_majority,
}

# Method registries
BACKSWING_METHODS = {
    'argmin_y': _argmin_y,
    'argmin_xy': _argmin_xy_sum,
    'velocity_zero': _velocity_zero_crossing,
}

CONTACT_METHODS = {
    'argmax_y': _argmax_y,
    'argmax_xy': _argmax_xy_sum,
    'argmax_x': _argmax_x,
}


def get_all_method_picks(wrist_xy, direction='backswing'):
    """Run all individual methods and return their picks.

    Parameters
    ----------
    wrist_xy : np.ndarray
        (T, 2) array of wrist x, y coordinates within the refinement window.
    direction : str
        'backswing' or 'contact'

    Returns
    -------
    dict : method_name -> frame index (relative to window start), or None
    """
    methods = BACKSWING_METHODS if direction == 'backswing' else CONTACT_METHODS
    picks = {}
    for name, fn in methods.items():
        result = fn(wrist_xy)
        picks[name] = result
    return picks


def refine_backswing_apex(wrist_xy, methods=None, voting='mean'):
    """Refine backswing apex using 2D ensemble.

    Parameters
    ----------
    wrist_xy : np.ndarray
        (T, 2) right wrist coordinates in the refinement window.
    methods : list of str, optional
        Which methods to use. Default: all available.
        Options: 'argmin_y', 'argmin_xy', 'velocity_zero'
    voting : str
        How to combine picks: 'mean', 'median', 'majority'

    Returns
    -------
    dict with keys:
        ensemble_idx: int - voted frame index (relative to window)
        picks: dict - individual method picks
        voting: str - strategy used
        n_methods: int - number of methods that returned a value
    """
    if methods is None:
        methods = list(BACKSWING_METHODS.keys())

    picks = {}
    for name in methods:
        fn = BACKSWING_METHODS[name]
        result = fn(wrist_xy)
        picks[name] = result

    valid = [v for v in picks.values() if v is not None]
    if len(valid) == 0:
        return {'ensemble_idx': 0, 'picks': picks, 'voting': voting, 'n_methods': 0}

    vote_fn = VOTING_STRATEGIES[voting]
    ensemble_idx = vote_fn(valid)
    ensemble_idx = max(0, min(ensemble_idx, len(wrist_xy) - 1))

    return {
        'ensemble_idx': ensemble_idx,
        'picks': picks,
        'voting': voting,
        'n_methods': len(valid),
    }


def refine_contact_point(wrist_xy, methods=None, voting='mean'):
    """Refine contact/impact point using inverse 2D ensemble.

    Parameters
    ----------
    wrist_xy : np.ndarray
        (T, 2) right wrist coordinates in the contact search window.
    methods : list of str, optional
        Which methods to use. Default: all available.
        Options: 'argmax_y', 'argmax_xy', 'argmax_x'
    voting : str
        How to combine picks: 'mean', 'median', 'majority'

    Returns
    -------
    dict with same structure as refine_backswing_apex
    """
    if methods is None:
        methods = list(CONTACT_METHODS.keys())

    picks = {}
    for name in methods:
        fn = CONTACT_METHODS[name]
        result = fn(wrist_xy)
        picks[name] = result

    valid = [v for v in picks.values() if v is not None]
    if len(valid) == 0:
        return {'ensemble_idx': 0, 'picks': picks, 'voting': voting, 'n_methods': 0}

    vote_fn = VOTING_STRATEGIES[voting]
    ensemble_idx = vote_fn(valid)
    ensemble_idx = max(0, min(ensemble_idx, len(wrist_xy) - 1))

    return {
        'ensemble_idx': ensemble_idx,
        'picks': picks,
        'voting': voting,
        'n_methods': len(valid),
    }


# --- Score-hand swing validation ---

def find_score_hand(kps, scores, min_consecutive=30, max_gap=1, conf_threshold=0.3):
    """Detect follow-through sequences where ONE wrist is above the shoulder.

    Parameters
    ----------
    kps : np.ndarray
        (N, 17, 2) keypoint coordinates.
    scores : np.ndarray
        (N, 17) confidence scores.
    min_consecutive : int
        Minimum consecutive frames for a valid sequence.
    max_gap : int
        Maximum gap frames to bridge.
    conf_threshold : float
        Minimum confidence for relevant keypoints.

    Returns
    -------
    list of dict with keys: start_frame, end_frame, length, wrist_above ('left'|'right')
    """
    n = len(kps)
    idxs = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
    valid = np.ones(n, dtype=bool)
    for idx in idxs:
        valid &= scores[:, idx] >= conf_threshold

    l_above = (
        (kps[:, 9, 1] < kps[:, 5, 1]) &   # l_wrist above l_shoulder
        (kps[:, 9, 1] < kps[:, 7, 1]) &   # l_wrist above l_elbow
        (kps[:, 10, 1] > kps[:, 6, 1])    # r_wrist below r_shoulder
    )
    r_above = (
        (kps[:, 10, 1] < kps[:, 6, 1]) &  # r_wrist above r_shoulder
        (kps[:, 10, 1] < kps[:, 8, 1]) &  # r_wrist above r_elbow
        (kps[:, 9, 1] > kps[:, 5, 1])     # l_wrist below l_shoulder
    )

    label = np.zeros(n, dtype=int)
    label[l_above & valid] = 1
    label[r_above & valid] = 2

    sequences = []
    i = 0
    while i < n:
        cur = label[i]
        if cur > 0:
            start = i
            end = i
            while i < n:
                if label[i] == cur:
                    end = i
                    i += 1
                elif label[i] == 0:
                    bridged = False
                    for look in range(1, max_gap + 1):
                        if i + look < n and label[i + look] == cur:
                            i += look
                            bridged = True
                            break
                        elif i + look < n and label[i + look] != 0:
                            break
                    if not bridged:
                        break
                else:
                    break
            length = end - start + 1
            if length >= min_consecutive:
                sequences.append({
                    'start_frame': start, 'end_frame': end,
                    'length': length,
                    'wrist_above': 'left' if cur == 1 else 'right',
                })
        else:
            i += 1
    return sequences


def find_both_wrists_above(kps, scores, conf_threshold=0.5):
    """Find frames where BOTH wrists are above BOTH shoulders and elbows."""
    idxs = [5, 6, 7, 8, 9, 10]
    valid = np.ones(len(kps), dtype=bool)
    for idx in idxs:
        valid &= scores[:, idx] >= conf_threshold

    both = (
        (kps[:, 9, 1] < kps[:, 7, 1]) &
        (kps[:, 10, 1] < kps[:, 8, 1]) &
        (kps[:, 9, 1] < kps[:, 5, 1]) &
        (kps[:, 10, 1] < kps[:, 6, 1]) &
        valid
    )
    return np.where(both)[0]


def validate_peaks_with_score_hand(peak_frames, kps, scores,
                                   follow_through_window=120,
                                   backswing_window=60,
                                   min_consecutive=20,
                                   conf_threshold=0.3):
    """Validate detected peaks by checking for anatomical swing signatures.

    A peak is validated if:
    1. There is a score-hand follow-through within follow_through_window
       frames AFTER the peak, OR
    2. There are >=3 frames with both wrists above shoulders within
       backswing_window frames BEFORE the peak.
    """
    sequences = find_score_hand(kps, scores,
                                min_consecutive=min_consecutive,
                                conf_threshold=conf_threshold)
    both_above = find_both_wrists_above(kps, scores, conf_threshold=conf_threshold)

    validated = np.zeros(len(peak_frames), dtype=bool)
    for i, peak in enumerate(peak_frames):
        peak = int(peak)
        for seq in sequences:
            if seq['start_frame'] > peak and seq['start_frame'] - peak <= follow_through_window:
                validated[i] = True
                break
        if validated[i]:
            continue
        before = both_above[(both_above >= peak - backswing_window) & (both_above <= peak)]
        if len(before) >= 3:
            validated[i] = True
    return validated
