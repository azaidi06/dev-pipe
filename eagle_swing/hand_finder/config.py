"""Configuration and result dataclasses for hand-raise detection."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class HandFinderConfig:
    left_shoulder: int = 5
    right_shoulder: int = 6
    left_elbow: int = 7
    right_elbow: int = 8
    left_wrist: int = 9
    right_wrist: int = 10

    conf_threshold: float = 0.3
    non_raised_pct: float = 0.80
    min_raise_frames: int = 40
    crop_padding: int = 80

    plateau_savgol_window: int = 11
    plateau_savgol_poly: int = 2
    plateau_vel_percentile: float = 30.0
    plateau_min_frames: int = 10


@dataclass
class HandFinderResult:
    name: str
    hand_frames: List[Optional[Tuple[int, int]]]
    representative_frames: List[Optional[int]]
    raised_side: List[Optional[str]]
    crop_boxes: List[Optional[Tuple[int, int, int, int]]]
    classifications: List[str]
    contact_result: object
    finger_counts: List[Optional[int]] = field(default_factory=list)
    finger_confidences: List[Optional[float]] = field(default_factory=list)

    @property
    def n_found(self) -> int:
        return sum(1 for c in self.classifications if c == "scored_swing")

    @property
    def n_swings(self) -> int:
        return len(self.classifications)
