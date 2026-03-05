"""Pipeline wrapper: video ingest/upload handler references."""


def handle_upload(event, context):
    """Process an incoming video upload event."""
    raise NotImplementedError("Ingest handler requires AWS runtime context")
