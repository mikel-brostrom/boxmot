"""Export orphan outcomes and prune files even when a camera never restarts."""

from contextlib import ExitStack
from pathlib import Path

from boxmot.attributes.database import connect
from boxmot.attributes.streams import CameraJournal, CameraLease, OwnershipLost
from boxmot.engine.streaming.outputs import AttributeEventWriter, export_events, prune_outputs


def maintain_outputs(database_url_env: str, camera_id: str, output_dir: Path, retention_days: int) -> None:
    """Replay abandoned events under ownership, leaving active owners to export."""
    with connect(database_url_env) as db:
        pending_sql = """SELECT 1 FROM boxmot_attributes.events e JOIN boxmot_attributes.runs r USING(run_id)
            WHERE r.camera_id=%s AND e.delivered_at IS NULL LIMIT 1"""
        if db.execute(pending_sql, (camera_id,)).fetchone() is not None:
            try:
                lease = CameraLease(camera_id, database_url_env)
            except OwnershipLost:
                # A live worker owns export. It also performs hourly file cleanup.
                return
            with ExitStack() as resources:
                resources.callback(lease.close)
                journal = CameraJournal(lease)
                resources.callback(journal.close)
                writer = AttributeEventWriter(output_dir / camera_id, max_bytes=64 * 1024 * 1024, max_seconds=3600)
                resources.callback(writer.close)
                # Bound a maintenance Job; later invocations continue any backlog.
                for _ in range(250):
                    if not export_events(journal, writer):
                        break
        if db.execute(pending_sql, (camera_id,)).fetchone() is None:
            live = db.execute(
                "SELECT run_id FROM boxmot_attributes.runs WHERE camera_id=%s AND state='active'", (camera_id,)
            ).fetchall()
            prune_outputs(
                output_dir / camera_id, retention_days, live_sessions=frozenset(row["run_id"] for row in live)
            )
