"""Failure categories shared by visual attribute backends and scheduling."""


class AttributeAnalysisError(RuntimeError):
    """A batch failed permanently and must produce an error result."""


class TransientAttributeError(AttributeAnalysisError):
    """A transport or server failure may be retried by the session worker."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        """Carry a provider's retry delay separately from the persisted message."""
        super().__init__(message)
        self.retry_after_s = retry_after_s
