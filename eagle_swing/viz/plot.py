"""Visualization: swing metric plots and comparisons.

Merged from eagle_swing/plot.py and golf-pipeline-final/analyze/plots.py.
Uses matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..analysis.upper_body import compute_upper_metrics
from ..analysis.lower_body import compute_lower_metrics


def plot_upper_body_comparison(kps_list, scores_list=None, labels=None,
                               handedness="right", title="Upper Body Comparison"):
    """Compare upper-body metrics across multiple swings.

    Parameters
    ----------
    kps_list : list of (T, 17, 2) arrays
    scores_list : list of (T, 17) arrays or None
    labels : list of str
    """
    n = len(kps_list)
    if scores_list is None:
        scores_list = [None] * n
    if labels is None:
        labels = [f"Swing {i + 1}" for i in range(n)]

    metrics_list = [compute_upper_metrics(k, s, handedness=handedness) for k, s in zip(kps_list, scores_list)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    for i, (m, lbl) in enumerate(zip(metrics_list, labels)):
        axes[0].plot(m["shoulder_angle_deg"], label=lbl, color=colors[i], alpha=0.85)
        axes[1].plot(m["shoulder_delta_deg"], label=lbl, color=colors[i], alpha=0.85)
        axes[2].plot(m["wrist_y_lead"], label=lbl, color=colors[i], alpha=0.85)
        axes[3].plot(m["wrist_y_trail"], label=lbl, color=colors[i], alpha=0.85)

    axes[0].set_ylabel("shoulder angle (deg)")
    axes[1].set_ylabel("shoulder delta (deg)")
    axes[2].set_ylabel("lead wrist y")
    axes[3].set_ylabel("trail wrist y")
    axes[3].set_xlabel("frame")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_lower_body_comparison(kps_list, scores_list=None, labels=None,
                               handedness="right", title="Lower Body Comparison"):
    """Compare lower-body metrics across multiple swings."""
    n = len(kps_list)
    if scores_list is None:
        scores_list = [None] * n
    if labels is None:
        labels = [f"Swing {i + 1}" for i in range(n)]

    metrics_list = [compute_lower_metrics(k, s, handedness=handedness) for k, s in zip(kps_list, scores_list)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    for i, (m, lbl) in enumerate(zip(metrics_list, labels)):
        axes[0].plot(m["hip_rotation_deg"], label=lbl, color=colors[i], alpha=0.85)
        axes[1].plot(m["lead_hip_lateral_shift"], label=lbl, color=colors[i], alpha=0.85)
        axes[2].plot(m["lead_knee_flexion"], label=lbl, color=colors[i], alpha=0.85)
        axes[3].plot(m["knee_hip_ratio"], label=lbl, color=colors[i], alpha=0.85)

    axes[0].set_ylabel("hip rotation (deg)")
    axes[1].set_ylabel("lateral shift (px)")
    axes[2].set_ylabel("lead knee flexion (deg)")
    axes[3].set_ylabel("knee/hip ratio")
    axes[3].set_xlabel("frame")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_swing_metrics(metrics_dict, keys=None, title="Swing Metrics"):
    """Plot selected metric time series from compute_swing_metrics output."""
    if keys is None:
        keys = list(metrics_dict.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.plot(metrics_dict[k])
        ax.set_ylabel(k)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("frame")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
