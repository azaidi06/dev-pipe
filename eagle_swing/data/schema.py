"""Dataclasses for swing keypoint data and detection results."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SwingMeta:
    """Metadata extracted from a pkl file."""
    pkl_path: str
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    n_pkl_frames: int = 0
    video_path: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class KeypointData:
    """Unified keypoint array container.

    Attributes:
        keypoints: (N, 17, 2) x/y coordinates per frame.
        scores: (N, 17) confidence scores per frame.
        meta: Associated metadata.
    """
    keypoints: np.ndarray
    scores: np.ndarray
    meta: SwingMeta

    @property
    def raw(self) -> np.ndarray:
        """(N, 17, 3) array with confidence as third channel."""
        return np.concatenate([self.keypoints, self.scores[..., None]], axis=2)

    def __len__(self):
        return len(self.keypoints)

    def frame(self, idx: int) -> np.ndarray:
        """Single frame as (17, 3) array [x, y, score]."""
        return np.column_stack((self.keypoints[idx], self.scores[idx]))

    def slice(self, start: int = 0, end: Optional[int] = None) -> "KeypointData":
        """Return a new KeypointData for the given frame range."""
        return KeypointData(
            keypoints=self.keypoints[start:end].copy(),
            scores=self.scores[start:end].copy(),
            meta=self.meta,
        )
