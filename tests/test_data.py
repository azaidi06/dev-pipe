"""Tests for the data module."""

import numpy as np
import pytest

from eagle_swing.data.loader import load_pkl, interpolate_and_smooth, discover_sessions
from eagle_swing.data.schema import KeypointData, SwingMeta


class TestLoadPkl:
    def test_pipeline_format(self, synthetic_pkl):
        kd = load_pkl(synthetic_pkl)
        assert isinstance(kd, KeypointData)
        assert kd.keypoints.shape == (600, 17, 2)
        assert kd.scores.shape == (600, 17)
        assert kd.meta.fps == 60.0
        assert kd.meta.n_pkl_frames == 600

    def test_eagle_format(self, synthetic_eagle_pkl):
        kd = load_pkl(synthetic_eagle_pkl)
        assert isinstance(kd, KeypointData)
        assert kd.keypoints.shape == (100, 17, 2)
        assert kd.scores.shape == (100, 17)
        assert kd.meta.fps is None  # no __meta__

    def test_len(self, synthetic_pkl):
        kd = load_pkl(synthetic_pkl)
        assert len(kd) == 600

    def test_frame(self, synthetic_pkl):
        kd = load_pkl(synthetic_pkl)
        frame = kd.frame(0)
        assert frame.shape == (17, 3)

    def test_slice(self, synthetic_pkl):
        kd = load_pkl(synthetic_pkl)
        sliced = kd.slice(10, 20)
        assert len(sliced) == 10

    def test_raw(self, synthetic_pkl):
        kd = load_pkl(synthetic_pkl)
        raw = kd.raw
        assert raw.shape == (600, 17, 3)


class TestInterpolateSmooth:
    def test_preserves_shape(self, synthetic_pkl):
        kd = load_pkl(synthetic_pkl)
        smoothed = interpolate_and_smooth(kd)
        assert smoothed.keypoints.shape == kd.keypoints.shape


class TestSchema:
    def test_swing_meta(self):
        meta = SwingMeta(pkl_path="/tmp/test.pkl")
        assert meta.pkl_path == "/tmp/test.pkl"
        assert meta.fps is None

    def test_keypoint_data(self):
        kps = np.zeros((10, 17, 2))
        scores = np.ones((10, 17))
        meta = SwingMeta(pkl_path="/tmp/test.pkl", n_pkl_frames=10)
        kd = KeypointData(keypoints=kps, scores=scores, meta=meta)
        assert len(kd) == 10
