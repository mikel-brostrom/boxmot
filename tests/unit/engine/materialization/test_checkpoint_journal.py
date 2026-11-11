"""Durability and bounded metadata writes for per-image checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import boxmot.engine.materialization.state as state_module
from boxmot.engine.materialization.plan import BuildPlan, PublishOptions, StagePlan
from boxmot.engine.materialization.state import MaterializationStateStore, StateError


def _plan(tmp_path: Path) -> BuildPlan:
    """Create a local plan without loading inference components."""

    detect = StagePlan.create("detect")
    embed = StagePlan.create("embed", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    finalize = StagePlan.create("finalize", depends_on=("embed",), upstream_fingerprints=(embed.fingerprint,))
    return BuildPlan.create(
        build_root=tmp_path,
        dataset_name="journal-test",
        box_type="aabb",
        source_fingerprint="a" * 64,
        publish=PublishOptions(embeddings=True),
        stages=(detect, embed, finalize),
    )


def _running_store(plan: BuildPlan) -> MaterializationStateStore:
    """Initialize a fresh store with embedding generation in progress."""

    store = MaterializationStateStore(plan.state_path)
    store.initialize(plan)
    store.begin("detect")
    store.complete("detect")
    store.begin("embed")
    return store


def _journal_path(plan: BuildPlan) -> Path:
    """Resolve the checkpoint journal beside the atomic state snapshot."""

    return plan.state_path.with_suffix(".jsonl")


def _append_bytes(path: Path, payload: bytes) -> None:
    """Simulate persisted journal bytes from a terminated writer."""

    with path.open("ab") as stream:
        stream.write(payload)


def test_journal_recovers_images_and_latest_hashes_without_snapshot_rewrites(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    snapshot = plan.state_path.read_bytes()
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    first_state = store.state
    store.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})
    store.record_shard("embed", "00000", hashes={"embeddings": "c" * 64})

    assert plan.state_path.read_bytes() == snapshot
    assert first_state.by_name["embed"].completed_shards == ("00000",)
    assert first_state.by_name["embed"].shard_hashes == (("00000", "embeddings", "a" * 64),)
    assert store.completed_shards("embed") == frozenset({"00000", "00001"})
    assert store.completed_shard_count("embed") == 2
    returned_hashes = store.shard_hashes("embed", "00000")
    returned_hashes["embeddings"] = "d" * 64
    assert store.shard_hashes("embed", "00000") == {"embeddings": "c" * 64}

    resumed = MaterializationStateStore(plan.state_path)
    state = resumed.initialize(plan)

    assert state.by_name["embed"].status == "pending"
    assert resumed.completed_shards("embed") == frozenset({"00000", "00001"})
    assert resumed.completed_shard_count("embed") == 2
    assert resumed.shard_hashes("embed", "00000") == {"embeddings": "c" * 64}
    assert resumed.shard_hashes("embed", "00001") == {"embeddings": "b" * 64}
    assert resumed.shard_hashes("embed", "00002") == {}
    snapshot_data = json.loads(plan.state_path.read_text(encoding="utf-8"))
    assert snapshot_data["journal_offset"] == _journal_path(plan).stat().st_size


def test_checkpoint_metadata_grows_linearly_with_image_count(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    snapshot = plan.state_path.read_bytes()
    sizes = []

    for index in range(128):
        store.record_shard("embed", f"{index:05d}", hashes={"embeddings": "a" * 64})
        if index in {63, 127}:
            sizes.append(_journal_path(plan).stat().st_size)

    assert plan.state_path.read_bytes() == snapshot
    assert sizes[1] == 2 * sizes[0]
    records = _journal_path(plan).read_bytes().splitlines()
    assert len(records) == 128
    assert max(map(len, records)) < 128
    assert store.completed_shard_count("embed") == 128


def test_checkpoint_updates_and_lookups_do_not_materialize_full_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)

    def unexpected_snapshot(_store: MaterializationStateStore) -> None:
        pytest.fail("per-image operations must not rebuild the full checkpoint snapshot")

    monkeypatch.setattr(MaterializationStateStore, "state", property(unexpected_snapshot))
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    store.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})

    assert store.completed_shard_count("embed") == 2
    assert store.completed_shards("embed") == frozenset({"00000", "00001"})
    assert store.shard_hashes("embed", "00001") == {"embeddings": "b" * 64}
    assert not store.is_completed("embed")


def test_discard_replays_without_restoring_old_checkpoint(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    store.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})
    store.complete("embed", artifacts=("embeddings",))
    snapshot = plan.state_path.read_bytes()
    store.discard_shard("embed", "00000")

    assert plan.state_path.read_bytes() == snapshot
    assert store.state.by_name["embed"].status == "running"
    assert store.state.by_name["embed"].artifacts == ()

    resumed = MaterializationStateStore(plan.state_path)
    state = resumed.initialize(plan)

    assert state.by_name["embed"].status == "pending"
    assert state.by_name["embed"].completed_at is None
    assert state.by_name["embed"].artifacts == ()
    assert resumed.completed_shards("embed") == frozenset({"00001"})
    assert resumed.shard_hashes("embed", "00000") == {}
    assert resumed.shard_hashes("embed", "00001") == {"embeddings": "b" * 64}


def test_replace_shards_preserves_only_replacement_and_subsequent_images(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    store.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})
    store.replace_shards("embed", {"00001": {"embeddings": "c" * 64}})
    store.record_shard("embed", "00002", hashes={"embeddings": "d" * 64})

    resumed = MaterializationStateStore(plan.state_path)
    resumed.initialize(plan)

    assert resumed.completed_shards("embed") == frozenset({"00001", "00002"})
    assert resumed.shard_hashes("embed", "00000") == {}
    assert resumed.shard_hashes("embed", "00001") == {"embeddings": "c" * 64}
    assert resumed.shard_hashes("embed", "00002") == {"embeddings": "d" * 64}


def test_invalidation_does_not_replay_obsolete_images(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    store.complete("embed")
    store.begin("finalize")
    store.record_shard("finalize", "00000", hashes={"manifest": "b" * 64})
    store.invalidate(("embed", "finalize"))
    store.begin("embed")
    store.record_shard("embed", "00001", hashes={"embeddings": "c" * 64})

    resumed = MaterializationStateStore(plan.state_path)
    state = resumed.initialize(plan)

    assert state.by_name["detect"].status == "completed"
    assert resumed.completed_shards("embed") == frozenset({"00001"})
    assert resumed.shard_hashes("embed", "00000") == {}
    assert resumed.completed_shards("finalize") == frozenset()
    assert resumed.shard_hashes("finalize", "00000") == {}


@pytest.mark.parametrize("transition", ["complete", "fail"])
def test_stage_transition_snapshots_all_recorded_images(tmp_path: Path, transition: str) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    if transition == "complete":
        store.complete("embed")
    else:
        store.fail("embed", "inference failed")

    payload = json.loads(plan.state_path.read_text(encoding="utf-8"))
    assert payload["journal_offset"] == _journal_path(plan).stat().st_size
    assert next(stage for stage in payload["stages"] if stage["name"] == "embed")["completed_shards"] == ["00000"]
    resumed = MaterializationStateStore(plan.state_path)
    state = resumed.initialize(plan)
    assert state.by_name["embed"].status == ("completed" if transition == "complete" else "failed")
    assert resumed.shard_hashes("embed", "00000") == {"embeddings": "a" * 64}


@pytest.mark.parametrize("tail", [b'["embed","00001",', b"\xff\xfe", b'["embed","00001",{}]'])
def test_resume_truncates_uncommitted_tail_before_new_appends(tmp_path: Path, tail: bytes) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    journal = _journal_path(plan)
    committed = journal.read_bytes()
    _append_bytes(journal, tail)

    resumed = MaterializationStateStore(plan.state_path)
    resumed.initialize(plan)

    assert journal.read_bytes() == committed
    assert resumed.completed_shards("embed") == frozenset({"00000"})
    resumed.begin("embed")
    resumed.record_shard("embed", "00002", hashes={"embeddings": "c" * 64})
    again = MaterializationStateStore(plan.state_path)
    again.initialize(plan)
    assert again.completed_shards("embed") == frozenset({"00000", "00002"})


@pytest.mark.parametrize(
    "record",
    [
        b"not-json\n",
        b"\xff\xfe\n",
        b"{}\n",
        b'["embed","00000"]\n',
        b'["unknown","00000",{}]\n',
        b'["embed","bad-id",{}]\n',
        b'["embed","00000",[]]\n',
        b'["embed","00000",{"embeddings":"bad-digest"}]\n',
        ('["embed","00000",{"invalid/name":"' + "a" * 64 + '"}]\n').encode(),
    ],
)
def test_resume_rejects_corrupt_complete_journal_records(tmp_path: Path, record: bytes) -> None:
    plan = _plan(tmp_path)
    _running_store(plan)
    _append_bytes(_journal_path(plan), record)

    with pytest.raises(StateError):
        MaterializationStateStore(plan.state_path).initialize(plan)


@pytest.mark.parametrize("offset", [-1, True, 1.5, 1, 100000])
def test_resume_rejects_invalid_snapshot_journal_offset(tmp_path: Path, offset: object) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    payload = json.loads(plan.state_path.read_text(encoding="utf-8"))
    payload["journal_offset"] = offset
    plan.state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError):
        MaterializationStateStore(plan.state_path).initialize(plan)


def test_resume_rejects_missing_journal_with_consumed_records(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    store.complete("embed")
    _journal_path(plan).unlink()

    with pytest.raises(StateError):
        MaterializationStateStore(plan.state_path).initialize(plan)


def test_orphaned_journal_cannot_be_adopted_by_a_new_build(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    plan.state_path.unlink()
    journal = _journal_path(plan).read_bytes()

    with pytest.raises(StateError):
        MaterializationStateStore(plan.state_path).initialize(plan)

    assert not plan.state_path.exists()
    assert _journal_path(plan).read_bytes() == journal


@pytest.mark.parametrize("operation", ["record", "discard"])
def test_failed_journal_sync_preserves_memory_and_durable_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    journal = _journal_path(plan)
    before_journal = journal.read_bytes()
    before_state = store.state
    real_fsync = state_module.os.fsync
    attempts = 0

    def fail_first_sync(fd: int) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated checkpoint sync failure")
        real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(state_module.os, "fsync", fail_first_sync)
        with pytest.raises(OSError, match="simulated checkpoint sync failure"):
            if operation == "record":
                store.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})
            else:
                store.discard_shard("embed", "00000")

    assert store.state == before_state
    assert journal.read_bytes() == before_journal
    assert store.completed_shards("embed") == frozenset({"00000"})
    assert store.shard_hashes("embed", "00000") == {"embeddings": "a" * 64}
    resumed = MaterializationStateStore(plan.state_path)
    resumed.initialize(plan)
    assert resumed.completed_shards("embed") == frozenset({"00000"})
    resumed.begin("embed")
    resumed.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})
    again = MaterializationStateStore(plan.state_path)
    again.initialize(plan)
    assert again.completed_shards("embed") == frozenset({"00000", "00001"})


@pytest.mark.parametrize("rollback_failure", ["fsync", "truncate"])
def test_uncertain_journal_rollback_blocks_writes_until_reinitialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback_failure: str
) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    before_state = store.state
    before_snapshot = plan.state_path.read_bytes()
    journal = _journal_path(plan)
    real_open = Path.open

    def failed_sync(_fd: int) -> None:
        raise OSError("simulated persistent sync failure")

    def failed_truncate(_size: int | None = None) -> None:
        raise OSError("simulated rollback truncation failure")

    with monkeypatch.context() as patch:
        if rollback_failure == "truncate":

            def open_with_failed_truncate(path: Path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                if path == journal and args and args[0] == "a+b":
                    patch.setattr(stream, "truncate", failed_truncate)
                return stream

            patch.setattr(Path, "open", open_with_failed_truncate)
        patch.setattr(state_module.os, "fsync", failed_sync)
        with pytest.raises(OSError, match="simulated persistent sync failure") as failure:
            store.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})

    assert isinstance(failure.value.__cause__, OSError)
    assert failure.value.__cause__ is not failure.value
    cause_message = "rollback truncation" if rollback_failure == "truncate" else "persistent sync"
    assert cause_message in str(failure.value.__cause__)

    assert store.state == before_state
    with pytest.raises(StateError):
        store.record_shard("embed", "00002", hashes={"embeddings": "c" * 64})
    with pytest.raises(StateError):
        store.fail("embed", "checkpoint failed")
    assert store.state == before_state
    assert plan.state_path.read_bytes() == before_snapshot

    # A full unacknowledged line can be replayed; a successfully truncated line
    # stays absent. Neither outcome may be hidden behind a failed-stage snapshot.
    store.initialize(plan)
    expected = {"00000", "00001"} if rollback_failure == "truncate" else {"00000"}
    assert store.state.by_name["embed"].status == "pending"
    assert store.completed_shards("embed") == frozenset(expected)
    store.begin("embed")
    store.record_shard("embed", "00002", hashes={"embeddings": "c" * 64})
    resumed = MaterializationStateStore(plan.state_path)
    resumed.initialize(plan)
    assert resumed.completed_shards("embed") == frozenset(expected | {"00002"})


@pytest.mark.parametrize("transition", ["replace", "invalidate", "complete"])
def test_failed_atomic_snapshot_restores_state_before_subsequent_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    plan = _plan(tmp_path)
    store = _running_store(plan)
    store.record_shard("embed", "00000", hashes={"embeddings": "a" * 64})
    store.record_shard("embed", "00001", hashes={"embeddings": "b" * 64})
    before_state = store.state
    before_snapshot = plan.state_path.read_bytes()

    def failed_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated atomic snapshot failure")

    with monkeypatch.context() as patch:
        patch.setattr(state_module.os, "replace", failed_replace)
        with pytest.raises(OSError, match="simulated atomic snapshot failure"):
            if transition == "replace":
                store.replace_shards("embed", {"00001": {"embeddings": "d" * 64}})
            elif transition == "invalidate":
                store.invalidate(("embed", "finalize"))
            else:
                store.complete("embed", artifacts=("embeddings",))

    assert store.state == before_state
    assert plan.state_path.read_bytes() == before_snapshot
    assert store.completed_shards("embed") == frozenset({"00000", "00001"})
    assert store.shard_hashes("embed", "00000") == {"embeddings": "a" * 64}
    assert store.shard_hashes("embed", "00001") == {"embeddings": "b" * 64}
    store.record_shard("embed", "00002", hashes={"embeddings": "c" * 64})

    resumed = MaterializationStateStore(plan.state_path)
    resumed.initialize(plan)
    assert resumed.completed_shards("embed") == frozenset({"00000", "00001", "00002"})
    assert resumed.shard_hashes("embed", "00000") == {"embeddings": "a" * 64}
    assert resumed.shard_hashes("embed", "00001") == {"embeddings": "b" * 64}
    assert resumed.shard_hashes("embed", "00002") == {"embeddings": "c" * 64}
