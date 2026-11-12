"""Opt-in PostgreSQL fixtures shared by attribute and engine contracts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def attribute_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Use a test database and remove only runs created by this test."""
    url = os.environ.get("BOXMOT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set BOXMOT_TEST_POSTGRES_URL to run PostgreSQL attribute tests.")
    import psycopg
    from psycopg.rows import dict_row

    from boxmot.attributes.spool import AttributeSpool

    monkeypatch.setenv("BOXMOT_ATTRIBUTES_DATABASE_URL", url)
    run_ids: list[str] = []
    initialize = AttributeSpool.__init__

    def record_run(self: AttributeSpool, config: Any, run_id: str) -> None:
        """Track this test's run IDs without affecting concurrent pytest workers."""
        run_ids.append(run_id)
        initialize(self, config, run_id)

    monkeypatch.setattr(AttributeSpool, "__init__", record_run)
    with psycopg.connect(url, autocommit=True, row_factory=dict_row, connect_timeout=5) as connection:
        try:
            yield connection
        finally:
            if run_ids:
                connection.execute("DELETE FROM boxmot_attributes.runs WHERE run_id = ANY(%s)", (run_ids,))
