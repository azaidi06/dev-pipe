"""Temporal alignment and derivative utilities.

Provides Savitzky-Golay based velocity/acceleration computation
for swing metric time series.
"""

import numpy as np
from scipy.signal import savgol_filter


def add_derivatives(data, window=7, poly=3, fps=60, axis=0):
    """Compute velocity and acceleration for a signal.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array. If 2D, only first 2 columns are used.
    window, poly : int
        Savgol filter parameters.
    fps : float
        Frames per second (used for delta).

    Returns
    -------
    vel, acc : np.ndarray
    """
    if data.ndim > 1:
        data = data[:, :2]
    dt = 1.0 / fps
    vel = savgol_filter(data, window_length=window, polyorder=poly, deriv=1, delta=dt, axis=axis)
    acc = savgol_filter(data, window_length=window, polyorder=poly, deriv=2, delta=dt, axis=axis)
    return vel, acc


def compute_metric_derivatives(metrics_dict, keys=None, window=7, poly=3, fps=60):
    """Compute velocity/acceleration for selected metrics in-place.

    Adds '{key}_vel' and '{key}_acc' entries to the dict.
    """
    if keys is None:
        keys = list(metrics_dict.keys())
    for k in keys:
        if k not in metrics_dict:
            continue
        data = metrics_dict[k]
        if not isinstance(data, np.ndarray) or len(data) < window:
            continue
        vel, acc = add_derivatives(data, window, poly, fps)
        metrics_dict[f"{k}_vel"] = vel
        metrics_dict[f"{k}_acc"] = acc
    return metrics_dict
