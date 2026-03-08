"""Configuration and result dataclasses for swing detection."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Config:
    # Smoothing
    savgol_window: int = 9
    savgol_poly: int = 3
    coarse_window: int = 61
    coarse_poly: int = 3
    # Peak detection
    peak_prominence: int = 300
    peak_distance: int = 300
    look_behind: int = 60
    look_ahead: int = 30
    search_back: int = 350
    refine_window: int = 15
    # Post-processing filters
    min_swing_gap: int = 300
    end_of_video_pct: float = 0.03
    xy_outlier_mad_thresh: float = 3.0
    xy_outlier_min_peaks: int = 3
    xy_outlier_mad_floor: int = 50
    xoff_mad_thresh: float = 3.0
    xoff_mad_floor: int = 60
    # Keypoint constants
    left_wrist: int = 9
    right_wrist: int = 10
    left_shoulder: int = 5
    right_shoulder: int = 6
    conf_threshold: float = 0.3
    # Contact detection
    contact_search_min: int = 10
    contact_search_max: int = 90
    contact_savgol_window: int = 5
    contact_savgol_poly: int = 2
    contact_refine_window: int = 10
    # Problem flagging
    min_expected_swings: int = 2
    max_expected_swings: int = 15
    low_conf_window: int = 5
    low_conf_threshold: float = 0.4
    close_gap_seconds: float = 8.0
    flag_mad_threshold: float = 2.0
    # CSV export
    downswing_outlier_frames: int = 40


@dataclass
class DetectionResult:
    name: str
    peak_frames: np.ndarray
    smoothed: np.ndarray
    combined: np.ndarray
    fps: float
    total_frames: int
    filter_log: list = field(default_factory=list)
    pkl_data: dict = field(default=None, repr=False)
    pkl_path: str = ""
    mov_path: str = ""

    @property
    def n_swings(self):
        return len(self.peak_frames)


@dataclass
class ContactResult:
    name: str
    contact_frames: np.ndarray
    backswing_result: DetectionResult
    smoothed: np.ndarray
    filter_log: list = field(default_factory=list)

    @property
    def n_contacts(self):
        return int(np.sum(self.contact_frames >= 0))

    @property
    def valid_contact_frames(self):
        return self.contact_frames[self.contact_frames >= 0]
