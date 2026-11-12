"""Failure categories shared by visual attribute backends and scheduling."""


class AttributeAnalysisError(RuntimeError):
    """A batch failed permanently and must produce an error result."""


class TransientAttributeError(AttributeAnalysisError):
    """A transport or server failure may be retried by the session worker."""
