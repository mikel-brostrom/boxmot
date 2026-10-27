"""Camera evidence belongs to a build and survives independent workflow runs."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from boxmot.engine.calibration import cmc
from boxmot.engine.eval.cmc import prepare_eval_cmc, prepare_tuning_cmc
from boxmot.engine.materialization.cmc import materialize_cmc
from boxmot.trackers import TrackerSpec
from tests.unit.engine.calibration.test_cmc_parallel import _args
from tests.unit.engine.calibration.test_cmc_parallel import camera_build as camera_build


def test_independent_workflows_reuse_materialized_camera_evidence(camera_build, tmp_path, monkeypatch):
    args = _args(camera_build, 1)
    args.cmc_policy = "auto"
    canonical = {name: (camera_build / name).read_bytes() for name in ("manifest.json", "_SUCCESS")}
    signature = cmc.cmc_build_signature(camera_build)
    first = cmc.prepare_cmc_candidates(args, {}, tmp_path / "calibration")

    def forbidden(*args, **kwargs):
        pytest.fail("A materialized CMC hit must not estimate motion or decode images.")

    monkeypatch.setattr(cmc, "_prepare_cmc_candidates_uncached", forbidden)
    monkeypatch.setattr(cmc, "_bgr_image", forbidden)
    args.sequence_workers = 7
    spec = TrackerSpec("botsort", per_class=True, class_ids=(0, 1))
    evaluation = prepare_eval_cmc(args, spec, tmp_path / "eval")
    tuning = prepare_tuning_cmc(args, spec, tmp_path / "tune")
    assert evaluation == tuning
    assert first.report_path.read_bytes() == evaluation.report
    assert first.warps_path.read_bytes() == evaluation.warps
    assert first.policy == evaluation.policy
    assert len(list((camera_build / "cmc").glob("*/_SUCCESS"))) == 1
    assert cmc.cmc_build_signature(camera_build) == signature
    assert {name: (camera_build / name).read_bytes() for name in canonical} == canonical
    # Run directories own copies, so edits cannot poison the materialization.
    first.warps_path.write_bytes(b"changed run output")
    assert prepare_eval_cmc(args, spec, tmp_path / "next").warps == evaluation.warps


@pytest.mark.parametrize("change", ("warps", "marker", "image", "detections", "settings", "subset"))
def test_invalid_or_different_camera_inputs_require_new_analysis(camera_build, tmp_path, monkeypatch, change):
    args = _args(camera_build, 1)
    cmc.prepare_cmc_candidates(args, {}, tmp_path / "first")
    entry = next(path.parent for path in (camera_build / "cmc").glob("*/_SUCCESS"))
    options = {}
    if change == "warps":
        (entry / "warps.npz").write_bytes(b"broken archive")
    elif change == "marker":
        (entry / "_SUCCESS").unlink()
    elif change == "image":
        path = next((camera_build / "images").glob("*.png"))
        path.write_bytes(path.read_bytes())
    elif change == "detections":
        path = next((camera_build / "instances").glob("*.parquet"))
        path.write_bytes(path.read_bytes())
    elif change == "settings":
        options = {"sof_target_width": 96}
    else:
        args.sequence_names = ("z-static",)
    calls = []
    original = cmc._prepare_cmc_candidates_uncached

    def generate(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(cmc, "_prepare_cmc_candidates_uncached", generate)
    cmc.prepare_cmc_candidates(args, options, tmp_path / "second")
    assert calls == [True]
    assert len(list((camera_build / "cmc").glob("*/_SUCCESS"))) == (2 if change in {"settings", "subset"} else 1)
    assert not list((camera_build / "cmc").glob(".*.tmp-*"))


def _validate_storage(report: bytes, warps: bytes) -> tuple[bytes, bytes]:
    if report != b"report" or warps != b"warps":
        raise ValueError("Invalid camera evidence")
    return report, warps


def _concurrent_materialization(build: Path, barrier, outcomes) -> None:
    """Race independent processes through an actual per-key filesystem lock."""
    def generate(path: Path) -> None:
        with (build / "generations").open("a") as stream:
            stream.write("generated\n")
        (path / "analysis.json").write_bytes(b"report")
        (path / "warps.npz").write_bytes(b"warps")

    barrier.wait(timeout=30)
    entry = materialize_cmc(
        build, inputs={"selection": "same"}, analysis_schema="test",
        generate=generate, validate=_validate_storage,
    )
    outcomes.put(entry.reused)


def test_concurrent_callers_generate_once(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier, outcomes = context.Barrier(2), context.Queue()
    processes = [
        context.Process(target=_concurrent_materialization, args=(tmp_path, barrier, outcomes)) for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=40)
            assert process.exitcode == 0
        assert sorted(outcomes.get(timeout=3) for _ in processes) == [False, True]
        assert (tmp_path / "generations").read_text().splitlines() == ["generated"]
        assert len(list((tmp_path / "cmc").glob("*/_SUCCESS"))) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        outcomes.close()
        outcomes.join_thread()


def test_failed_generation_never_publishes_partial_evidence(tmp_path):
    def failed(path: Path) -> None:
        (path / "analysis.json").write_bytes(b"report")
        raise RuntimeError("interrupted estimator")

    with pytest.raises(RuntimeError, match="interrupted estimator"):
        materialize_cmc(tmp_path, inputs={}, analysis_schema="test", generate=failed, validate=_validate_storage)
    assert not list((tmp_path / "cmc").glob("*/_SUCCESS"))
    assert not list((tmp_path / "cmc").glob(".*.tmp-*"))


def test_cache_hit_does_not_require_a_writable_lock(tmp_path, monkeypatch):
    from boxmot.engine.materialization import cmc as storage

    def generate(path: Path) -> None:
        (path / "analysis.json").write_bytes(b"report")
        (path / "warps.npz").write_bytes(b"warps")

    materialize_cmc(tmp_path, inputs={}, analysis_schema="test", generate=generate, validate=_validate_storage)

    def forbidden(*args, **kwargs):
        pytest.fail("Valid persisted entries must be readable without write access.")

    monkeypatch.setattr(storage, "FileLock", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert materialize_cmc(
        tmp_path, inputs={}, analysis_schema="test", generate=forbidden, validate=_validate_storage,
    ).reused
