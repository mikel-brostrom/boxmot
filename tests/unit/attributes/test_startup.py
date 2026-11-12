"""Check resource ownership when PostgreSQL attribute sessions cannot start."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.session import AttributeEnricher
from boxmot.attributes.spool import AttributeSpool


class _Backend:
    """Record cleanup without issuing model requests during startup."""

    def __init__(self) -> None:
        self.closed = False

    async def extract(self, *_args: Any) -> dict[str, Any]:
        """Fail if startup unexpectedly submits inference work."""
        raise AssertionError("Startup must not issue an inference request.")

    async def aclose(self) -> None:
        """Record that the worker released its backend."""
        self.closed = True


def _config() -> AttributeConfig:
    """Use a valid profile without requiring a model server."""
    return AttributeConfig(model="fixture", prompt="Describe", schema={"type": "object"})


@pytest.fixture
def captured_spools(attribute_database: Any, monkeypatch: pytest.MonkeyPatch) -> list[AttributeSpool]:
    """Keep initialized connections visible after a constructor raises."""
    del attribute_database
    spools: list[AttributeSpool] = []
    initialize = AttributeSpool.__init__

    def capture(self: AttributeSpool, config: AttributeConfig, run_id: str) -> None:
        """Preserve the database fixture's run cleanup and record the connection."""
        initialize(self, config, run_id)
        spools.append(self)

    monkeypatch.setattr(AttributeSpool, "__init__", capture)
    return spools


@pytest.mark.parametrize("inside_event_loop", [False, True])
def test_thread_start_failure_closes_database_without_creating_backend(
    captured_spools: list[AttributeSpool], monkeypatch: pytest.MonkeyPatch, inside_event_loop: bool,
) -> None:
    """Thread exhaustion must not allocate an HTTP client, even in async callers."""
    created: list[_Backend] = []

    def factory(_config: AttributeConfig) -> _Backend:
        """Record any premature backend allocation."""
        backend = _Backend()
        created.append(backend)
        return backend

    def fail_start(_thread: Any) -> None:
        """Simulate inability to allocate the worker thread."""
        raise RuntimeError("Cannot start worker thread.")

    async def construct_in_loop() -> None:
        """Exercise startup from a caller that already owns an event loop."""
        AttributeEnricher(_config())

    monkeypatch.setattr("boxmot.attributes.session._BACKENDS", SimpleNamespace(resolve=lambda _name: factory))
    monkeypatch.setattr("boxmot.attributes.session.threading.Thread.start", fail_start)
    with pytest.raises(RuntimeError, match="Cannot start worker thread"):
        if inside_event_loop:
            asyncio.run(construct_in_loop())
        else:
            AttributeEnricher(_config())
    assert created == []
    assert len(captured_spools) == 1
    assert captured_spools[0]._db.closed


def test_backend_constructor_failure_closes_database(
    captured_spools: list[AttributeSpool], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend factory failure must not strand a PostgreSQL connection."""
    def fail_backend(_config: AttributeConfig) -> _Backend:
        """Reject initialization before a backend can be returned."""
        raise ValueError("Backend initialization failed.")

    monkeypatch.setattr("boxmot.attributes.session._BACKENDS", SimpleNamespace(resolve=lambda _name: fail_backend))
    with pytest.raises(RuntimeError, match="Attribute worker failed") as error:
        AttributeEnricher(_config())
    assert isinstance(error.value.__cause__, ValueError)
    assert captured_spools[0]._db.closed


def test_initial_worker_query_failure_closes_backend_and_database(
    captured_spools: list[AttributeSpool], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness must include the first DB query and clean up a failed worker."""
    backend = _Backend()

    def fail_claim(self: AttributeSpool) -> None:
        """Trigger a real PostgreSQL error inside the normal transaction guard."""
        with self.transaction():
            self._db.execute("SELECT 1 / 0")

    monkeypatch.setattr(AttributeSpool, "claim", fail_claim)
    with pytest.raises(RuntimeError, match="Attribute worker failed") as error:
        AttributeEnricher(_config(), backend=backend)
    assert "PostgreSQL operation failed" in str(error.value.__cause__)
    assert backend.closed
    assert captured_spools[0]._db.closed


def test_close_stops_worker_before_database_inspection(
    captured_spools: list[AttributeSpool], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database failure during shutdown must not keep the worker running."""
    backend = _Backend()
    session = AttributeEnricher(_config(), backend=backend)

    def unavailable(_spool: AttributeSpool) -> int:
        """Check that cancellation precedes the first shutdown query."""
        assert session._stop.is_set()
        raise RuntimeError("Database unavailable during shutdown.")

    monkeypatch.setattr(AttributeSpool, "pending_count", property(unavailable))
    with pytest.raises(RuntimeError, match="Database unavailable during shutdown"):
        session.close()
    assert backend.closed
    assert not session._thread.is_alive()
    assert captured_spools[0]._db.closed
