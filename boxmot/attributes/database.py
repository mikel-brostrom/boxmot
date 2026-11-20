"""Versioned PostgreSQL storage and explicit, additive schema upgrades."""

from __future__ import annotations

import os
from typing import Any

from boxmot.utils.config import ConfigurationError
from boxmot.utils.dependencies import require_extra

_SCHEMA_VERSION = "boxmot_attributes:1"

_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS boxmot_attributes;
COMMENT ON SCHEMA boxmot_attributes IS 'boxmot_attributes:1';
CREATE TABLE IF NOT EXISTS boxmot_attributes.runs (
    run_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    backend TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ticket BIGINT NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS boxmot_attributes.tracks (
    run_id TEXT NOT NULL REFERENCES boxmot_attributes.runs(run_id) ON DELETE CASCADE,
    generation BIGINT NOT NULL,
    track_id BIGINT NOT NULL,
    sequence_id TEXT,
    class_id BIGINT NOT NULL,
    last_seen BIGINT NOT NULL,
    last_sample BIGINT,
    observation JSONB NOT NULL,
    ever_sampled BOOLEAN NOT NULL DEFAULT FALSE,
    unavailable_reported BOOLEAN NOT NULL DEFAULT FALSE,
    latest JSONB,
    last_status TEXT,
    PRIMARY KEY (run_id, generation, track_id)
);
CREATE INDEX IF NOT EXISTS tracks_seen ON boxmot_attributes.tracks(run_id, generation, last_seen);
CREATE TABLE IF NOT EXISTS boxmot_attributes.jobs (
    run_id TEXT NOT NULL,
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    ticket BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    track_id BIGINT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    ready BOOLEAN NOT NULL DEFAULT FALSE,
    attempts INTEGER NOT NULL DEFAULT 0,
    available TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, id),
    FOREIGN KEY (run_id, generation, track_id)
        REFERENCES boxmot_attributes.tracks(run_id, generation, track_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS one_pending
    ON boxmot_attributes.jobs(run_id, generation, track_id) WHERE state = 'pending';
CREATE UNIQUE INDEX IF NOT EXISTS one_frozen
    ON boxmot_attributes.jobs(run_id, generation, track_id) WHERE state IN ('running', 'retry');
CREATE INDEX IF NOT EXISTS ready_jobs ON boxmot_attributes.jobs(run_id, state, ready, ticket);
CREATE TABLE IF NOT EXISTS boxmot_attributes.crops (
    run_id TEXT NOT NULL,
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    job_id BIGINT NOT NULL,
    sample JSONB NOT NULL,
    image BYTEA NOT NULL,
    PRIMARY KEY (run_id, id),
    FOREIGN KEY (run_id, job_id) REFERENCES boxmot_attributes.jobs(run_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS job_crops ON boxmot_attributes.crops(run_id, job_id, id);
CREATE TABLE IF NOT EXISTS boxmot_attributes.events (
    run_id TEXT NOT NULL REFERENCES boxmot_attributes.runs(run_id) ON DELETE CASCADE,
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMPTZ,
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS undelivered_events
    ON boxmot_attributes.events(run_id, id) WHERE delivered_at IS NULL;
CREATE TABLE IF NOT EXISTS boxmot_attributes.counters (
    run_id TEXT NOT NULL REFERENCES boxmot_attributes.runs(run_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    count BIGINT NOT NULL,
    PRIMARY KEY (run_id, status)
);
"""


_UPGRADE_SQL = """
ALTER TABLE boxmot_attributes.runs ADD COLUMN camera_id TEXT;
ALTER TABLE boxmot_attributes.runs ADD COLUMN owner_id TEXT;
ALTER TABLE boxmot_attributes.runs ADD COLUMN epoch BIGINT;
ALTER TABLE boxmot_attributes.runs ADD COLUMN state TEXT NOT NULL DEFAULT 'active';
ALTER TABLE boxmot_attributes.runs ADD COLUMN ended_at TIMESTAMPTZ;
ALTER TABLE boxmot_attributes.runs ADD COLUMN display_fields JSONB NOT NULL DEFAULT '[]';
ALTER TABLE boxmot_attributes.tracks ADD COLUMN observed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE boxmot_attributes.tracks ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE boxmot_attributes.jobs ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;
CREATE INDEX runs_camera ON boxmot_attributes.runs(camera_id, state);
CREATE INDEX tracks_observed ON boxmot_attributes.tracks(observed_at);
CREATE INDEX events_created ON boxmot_attributes.events(created_at);
CREATE TABLE boxmot_attributes.camera_leases (
    camera_id TEXT PRIMARY KEY,
    owner_id TEXT,
    epoch BIGINT NOT NULL DEFAULT 0,
    expires_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
COMMENT ON SCHEMA boxmot_attributes IS 'boxmot_attributes:2';
"""
SCHEMA_VERSION = "boxmot_attributes:2"


def connect(database_url_env: str) -> Any:
    """Open a bounded connection without exposing credentials in driver errors."""
    url = os.environ.get(database_url_env, "").strip()
    if not url:
        raise ConfigurationError(
            f"Attribute database environment variable {database_url_env!r} must be set and nonempty."
        )
    require_extra("attributes", purpose="PostgreSQL track attribute storage")
    import psycopg
    from psycopg.rows import dict_row

    try:
        return psycopg.connect(
            url,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=3,
            options="-c statement_timeout=3000 -c lock_timeout=2000",
            application_name="boxmot-stream",
            keepalives=1,
            keepalives_idle=2,
            keepalives_interval=1,
            keepalives_count=3,
            tcp_user_timeout=3000,
        )
    except psycopg.Error as exc:
        raise ConfigurationError(f"Attribute PostgreSQL connection failed ({type(exc).__name__}).") from None


def ensure_schema(connection: Any, *, upgrade: bool = False) -> None:
    """Create a new schema or explicitly upgrade v1, under the shared DDL lock."""
    with connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(1112496205, 1096045650)")
        row = connection.execute(
            "SELECT pg_catalog.obj_description(oid, 'pg_namespace') AS version "
            "FROM pg_catalog.pg_namespace WHERE nspname = 'boxmot_attributes'"
        ).fetchone()
        if row is None:
            connection.execute(_SCHEMA_SQL, prepare=False)
            connection.execute(_UPGRADE_SQL, prepare=False)
        elif row["version"] == _SCHEMA_VERSION and upgrade:
            connection.execute(_UPGRADE_SQL, prepare=False)
        elif row["version"] != SCHEMA_VERSION:
            raise ConfigurationError(
                "Attribute PostgreSQL schema version is unsupported. "
                "Stop existing clients and run 'boxmot stream-db upgrade' before restarting them."
            )


def upgrade_database(database_url_env: str = "BOXMOT_ATTRIBUTES_DATABASE_URL") -> None:
    """Upgrade without deleting existing runs, results, or crops."""
    with connect(database_url_env) as connection:
        with connection.transaction():
            # Index construction on existing history needs more time than a
            # live stream transaction. These settings expire with this upgrade.
            connection.execute("SET LOCAL statement_timeout = '10min'")
            connection.execute("SET LOCAL lock_timeout = '30s'")
            ensure_schema(connection, upgrade=True)
