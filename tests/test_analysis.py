"""Tests for the analysis module."""

import numpy as np
import pytest

from eagle_swing.analysis.core import (
    compute_swing_metrics,
    normalize_scale_center,
    interpolate_and_filter,
    resample_1d,
    find_swing_medoid,
    get_velocity,
)
from eagle_swing.analysis.normalization import (
    normalize_by_average_torso,
    center_by_average_torso,
    normalize_scale_center_robust,
    align_to_body_frame_static,
)
from eagle_swing.analysis.upper_body import compute_upper_metrics
from eagle_swing.analysis.lower_body import compute_lower_metrics
from eagle_swing.analysis.temporal import add_derivatives
from eagle_swing.utils import angle_2pts, angle_3pts


@pytest.fixture
def fake_kps():
    """Synthetic (60, 17, 3) keypoints with confidence."""
    rng = np.random.RandomState(42)
    kps = rng.rand(60, 17, 2) * 400 + 50
    scores = rng.rand(60, 17) * 0.5 + 0.5
    return np.concatenate([kps, scores[..., None]], axis=2)


@pytest.fixture
def fake_kps_2d():
    """Synthetic (60, 17, 2) keypoints only."""
    rng = np.random.RandomState(42)
    return rng.rand(60, 17, 2) * 400 + 50


class TestSwingMetrics:
    def test_returns_all_keys(self, fake_kps_2d):
        metrics = compute_swing_metrics(fake_kps_2d)
        expected_keys = [
            "Hand Depth (X)", "Hand Height (Y)", "Spine Angle",
            "Hip Sway (X)", "Head Sway (X)", "Head Height (Y)",
            "Lead Knee Angle", "Trail Knee Angle", "Right Elbow Angle",
            "Shoulder Tilt", "Hip Tilt", "Hand-Body Dist",
            "Shoulder Turn (Proxy)", "Hip Turn (Proxy)", "X-Factor (Stretch)",
            "Shoulder Rot. Vel.", "Hip Rot. Vel.",
        ]
        for k in expected_keys:
            assert k in metrics
            assert len(metrics[k]) == 60

    def test_angles_in_range(self, fake_kps_2d):
        metrics = compute_swing_metrics(fake_kps_2d)
        assert np.all(metrics["Lead Knee Angle"] >= 0)
        assert np.all(metrics["Lead Knee Angle"] <= 180)


class TestNormalization:
    def test_scale_center(self, fake_kps):
        normed = normalize_scale_center(fake_kps)
        assert normed.shape == fake_kps.shape

    def test_average_torso(self, fake_kps):
        normed = normalize_by_average_torso(fake_kps)
        assert normed.shape == fake_kps.shape

    def test_center_average_torso(self, fake_kps):
        normed = center_by_average_torso(fake_kps)
        assert normed.shape == fake_kps.shape

    def test_robust(self, fake_kps):
        normed = normalize_scale_center_robust(fake_kps)
        assert normed.shape == fake_kps.shape

    def test_body_frame(self, fake_kps):
        normed = align_to_body_frame_static(fake_kps)
        assert normed.shape == fake_kps.shape


class TestUpperBody:
    def test_metrics_keys(self, fake_kps_2d):
        m = compute_upper_metrics(fake_kps_2d)
        assert "wrist_y_lead" in m
        assert "shoulder_angle_deg" in m
        assert len(m["wrist_y_lead"]) == 60


class TestLowerBody:
    def test_metrics_keys(self, fake_kps_2d):
        m = compute_lower_metrics(fake_kps_2d)
        assert "hip_rotation_deg" in m
        assert "lead_knee_flexion" in m
        assert len(m["hip_rotation_deg"]) == 60


class TestTemporal:
    def test_derivatives(self):
        data = np.sin(np.linspace(0, 2 * np.pi, 60))
        vel, acc = add_derivatives(data)
        assert vel.shape == data.shape
        assert acc.shape == data.shape


class TestUtils:
    def test_angle_2pts(self):
        p1 = np.array([[0, 0], [1, 0]])
        p2 = np.array([[1, 0], [1, 1]])
        angles = angle_2pts(p1, p2)
        assert angles.shape == (2,)

    def test_angle_3pts(self):
        a = np.array([[0, 0], [0, 0]])
        b = np.array([[1, 0], [1, 0]])
        c = np.array([[1, 1], [0, 1]])
        angles = angle_3pts(a, b, c)
        assert angles.shape == (2,)
        assert np.all(angles >= 0)


class TestResampling:
    def test_resample_1d(self):
        arr = np.sin(np.linspace(0, 2 * np.pi, 50))
        resampled = resample_1d(arr, target=100)
        assert len(resampled) == 100

    def test_find_medoid(self):
        swings = [np.random.rand(10, 2) for _ in range(5)]
        idx = find_swing_medoid(swings)
        assert 0 <= idx < 5


class TestInterpolateFilter:
    def test_preserves_shape(self, fake_kps):
        filtered = interpolate_and_filter(fake_kps)
        assert filtered.shape == fake_kps.shape
