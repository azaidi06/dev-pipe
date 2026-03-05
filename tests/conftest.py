"""Shared test fixtures: synthetic pkl data and ymirza path discovery."""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eagle_swing.data.loader import DATA_ROOT


@pytest.fixture
def synthetic_pkl(tmp_path):
    """Create a synthetic pipeline-format pkl file with 600 frames.

    Returns the path to the pkl file.
    """
    n_frames = 600
    fps = 60.0
    data = {}
    rng = np.random.RandomState(42)

    for i in range(n_frames):
        kps = rng.rand(17, 2) * 500 + 100
        scores = rng.rand(17) * 0.5 + 0.5

        # Simulate a wrist dip (backswing) at frames 150 and 450
        for bs_frame in [150, 450]:
            dist = abs(i - bs_frame)
            if dist < 30:
                kps[9, 1] -= (30 - dist) * 3
                kps[10, 1] -= (30 - dist) * 3

        data[f"frame_{i}"] = {"keypoints": kps, "keypoint_scores": scores}

    data["__meta__"] = {
        "fps": fps,
        "total_frames": n_frames,
        "n_pkl_frames": n_frames,
        "width": 1920,
        "height": 1080,
    }

    pkl_path = tmp_path / "test_swing.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    return str(pkl_path)


@pytest.fixture
def synthetic_eagle_pkl(tmp_path):
    """Create a synthetic eagle-swing format pkl (no __meta__, dict keys)."""
    n_frames = 100
    rng = np.random.RandomState(123)
    data = {}
    for i in range(n_frames):
        data[f"frame_{i}"] = {
            "keypoints": rng.rand(17, 2) * 400 + 50,
            "keypoint_scores": rng.rand(17) * 0.6 + 0.4,
        }

    pkl_path = tmp_path / "eagle_test.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    return str(pkl_path)


@pytest.fixture
def ymirza_sessions():
    """Discover ymirza session directories (skip if not present)."""
    root = Path(DATA_ROOT)
    if not root.exists():
        pytest.skip(f"ymirza data not found at {DATA_ROOT}")
    sessions = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not sessions:
        pytest.skip("No session directories found")
    return sessions


@pytest.fixture
def ymirza_pkl_path(ymirza_sessions):
    """Return path to the first pkl file found in ymirza data."""
    root = Path(DATA_ROOT)
    for session in ymirza_sessions:
        session_dir = root / session
        pkls = list(session_dir.rglob("*.pkl"))
        if pkls:
            return str(pkls[0])
    pytest.skip("No pkl files found in ymirza data")
