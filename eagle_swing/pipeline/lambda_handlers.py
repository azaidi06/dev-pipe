"""Pipeline wrapper: Lambda handler entry points.

These are thin entry points that the Lambda runtime calls.
They import detection/analysis logic from the eagle_swing package.
"""


def detection_handler(event, context):
    """S3 pkl trigger -> detect backswings + contacts -> write results."""
    raise NotImplementedError("Lambda handler requires AWS runtime context")


def analysis_handler(event, context):
    """S3 detection results -> full analysis -> write results."""
    raise NotImplementedError("Lambda handler requires AWS runtime context")
