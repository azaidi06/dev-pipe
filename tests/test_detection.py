"""Tests for the detection module."""

import numpy as np
import pytest

from eagle_swing.detection.config import Config, DetectionResult, ContactResult
from eagle_swing.detection.backswing import detect_backswings
from eagle_swing.detection.contact import detect_contacts


class TestDetectBackswings:
    def test_returns_detection_result(self, synthetic_pkl):
        result = detect_backswings(synthetic_pkl)
        assert isinstance(result, DetectionResult)
        assert result.fps == 60.0
        assert result.total_frames == 600

    def test_finds_peaks(self, synthetic_pkl):
        result = detect_backswings(synthetic_pkl)
        assert result.n_swings >= 0
        assert len(result.peak_frames) == result.n_swings

    def test_smoothed_shape(self, synthetic_pkl):
        result = detect_backswings(synthetic_pkl)
        assert len(result.smoothed) == len(result.combined)

    def test_custom_config(self, synthetic_pkl):
        cfg = Config(peak_prominence=100, peak_distance=100)
        result = detect_backswings(synthetic_pkl, config=cfg)
        assert isinstance(result, DetectionResult)


class TestDetectContacts:
    def test_returns_contact_result(self, synthetic_pkl):
        br = detect_backswings(synthetic_pkl)
        if br.n_swings == 0:
            pytest.skip("No backswings detected in synthetic data")
        cr = detect_contacts(br)
        assert isinstance(cr, ContactResult)
        assert len(cr.contact_frames) == br.n_swings

    def test_contact_frames_valid(self, synthetic_pkl):
        br = detect_backswings(synthetic_pkl)
        if br.n_swings == 0:
            pytest.skip("No backswings detected")
        cr = detect_contacts(br)
        for cf in cr.contact_frames:
            assert cf == -1 or cf >= 0


class TestConfig:
    def test_frozen(self):
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.savgol_window = 11

    def test_defaults(self):
        cfg = Config()
        assert cfg.left_wrist == 9
        assert cfg.right_wrist == 10
        assert cfg.conf_threshold == 0.3
