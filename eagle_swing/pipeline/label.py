"""Pipeline wrapper: label_videos entry points.

Thin wrapper around the label_videos worker logic.
The actual GPU worker lives in golf-pipeline-final/label_videos/worker.py.
"""


def label_local(video_dir, skip_existing=True, batch_size=4):
    """Run local labeling (wrapper for worker.py --local mode).

    This is a placeholder that documents the interface.
    Actual implementation imports from the pipeline worker.
    """
    raise NotImplementedError(
        "Local labeling requires the full pipeline worker. "
        "Use: python -m label_videos.worker --local --video-dir <dir>"
    )
