"""Tests for the hand_finder module."""

import numpy as np
import pytest

from eagle_swing.hand_finder.config import HandFinderConfig, HandFinderResult
from eagle_swing.hand_finder.detect import (
    _find_all_runs,
    find_plateau,
    _check_raised,
    _check_non_raised,
)


class TestConfig:
    def test_frozen(self):
        cfg = HandFinderConfig()
        with pytest.raises(AttributeError):
            cfg.conf_threshold = 0.5

    def test_defaults(self):
        cfg = HandFinderConfig()
        assert cfg.left_wrist == 9
        assert cfg.min_raise_frames == 40


class TestFindAllRuns:
    def test_single_run(self):
        mask = np.array([False, True, True, True, False])
        runs = _find_all_runs(mask, min_len=2)
        assert len(runs) == 1
        assert runs[0] == (1, 4)

    def test_no_run(self):
        mask = np.array([True, False, True, False])
        runs = _find_all_runs(mask, min_len=2)
        assert len(runs) == 0

    def test_multiple_runs(self):
        mask = np.array([True, True, True, False, True, True, False])
        runs = _find_all_runs(mask, min_len=2)
        assert len(runs) == 2

    def test_empty(self):
        runs = _find_all_runs(np.array([], dtype=bool), min_len=1)
        assert runs == []


class TestFindPlateau:
    def test_flat_signal(self):
        wrist_y = np.ones(50) * 100
        plateau = find_plateau(wrist_y)
        assert plateau is not None
        assert plateau[1] - plateau[0] >= 10

    def test_short_signal(self):
        assert find_plateau(np.array([1, 2, 3])) is None


class TestCheckRaised:
    def test_wrist_above_shoulder(self):
        cfg = HandFinderConfig(conf_threshold=0.0)
        kps = np.zeros((5, 17, 2))
        scores = np.ones((5, 17))
        # Left wrist above left shoulder and elbow
        kps[:, 9, 1] = 50   # wrist y
        kps[:, 5, 1] = 100  # shoulder y
        kps[:, 7, 1] = 80   # elbow y
        mask = _check_raised(kps, scores, cfg, "LEFT")
        assert np.all(mask)

    def test_wrist_below_shoulder(self):
        cfg = HandFinderConfig(conf_threshold=0.0)
        kps = np.zeros((5, 17, 2))
        scores = np.ones((5, 17))
        kps[:, 9, 1] = 200  # wrist below
        kps[:, 5, 1] = 100
        kps[:, 7, 1] = 150
        mask = _check_raised(kps, scores, cfg, "LEFT")
        assert not np.any(mask)
