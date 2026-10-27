"""Evaluation decides CMC from its current sequences before tracking starts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from boxmot.engine.calibration import cmc as analysis
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy
from boxmot.engine.eval import cmc, evaluator
from boxmot.engine.eval.replay import ReplayResult
from boxmot.engine.eval.session import ReplaySession
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_cmc_signature
from tests.unit.engine.calibration.test_cmc_candidates import _Dataset, _sample, _source_build
from tests.unit.engine.eval.test_eval_display_workflow import _args


def _spec(**options: object) -> TrackerSpec:
    """Keep association changes separate from the frozen camera mask threshold."""
    return TrackerSpec(
        name="bytetrack",
        class_ids=(1,),
        class_names=((1, "person"),),
        options=tuple(sorted({"cmc_min_conf": 0.2, "cmc_method": "sof", **options}.items())),
    )


@pytest.fixture
def auto_inputs(tmp_path: Path):
    """Use sequence names that do not appear in any saved calibration artifact."""
    args = _args(tmp_path, cmc_policy="auto", seq_info={"unseen-moving": 4, "unseen-static": 4}, sequence_workers=1)
    _source_build(args.build_path)
    return args


@pytest.fixture
def stub_analysis(monkeypatch: pytest.MonkeyPatch):
    """Write small artifacts while retaining the exact requested input selection."""
    calls = []

    def prepare(args, baseline, output_dir, progress=None):
        calls.append((args.build_path, args.split, baseline.copy()))
        selected = sorted(args.sequence_names or args.seq_info)
        policy = SequenceCMCPolicy(sequences=tuple((name, "moving" in name) for name in selected))
        report = Path(output_dir) / "analysis.json"
        warps = Path(output_dir) / "warps.npz"
        spec = TrackerSpec(
            args.tracker, backend=args.tracker_backend, geometry=args.geometry, per_class=args.per_class,
            class_ids=args.tracker_class_ids, class_names=args.tracker_class_names,
            options=tuple(sorted(baseline.items())),
        )
        sequences, arrays = {}, {}
        image_root = str(args.build_path.parent / "images")
        for sequence in selected:
            count = args.seq_info[sequence]
            references = tuple(f"{sequence}/{index}.png" for index in range(count))
            for reference in references:
                image = Path(image_root) / reference
                image.parent.mkdir(parents=True, exist_ok=True)
                if not image.exists():
                    image.write_bytes(b"fixture pixels")
            arrays[f"{sequence}_warps"] = np.broadcast_to(np.eye(2, 3, dtype=np.float32), (count, 2, 3)).copy()
            arrays[f"{sequence}_indices"] = np.arange(count, dtype=np.int64)
            sequences[sequence] = {
                "frames": count, "enabled": "moving" in sequence,
                "sample_ids": [f"{sequence}/{index}" for index in range(count)],
                "image_refs": references, "source_image_root": image_root,
                "image_signature": analysis.cmc_image_signature(references, image_root),
                "archive_keys": {
                    "warps": f"{sequence}_warps", "runtime_warps": f"{sequence}_warps",
                    "frame_indices": f"{sequence}_indices",
                },
            }
        np.savez(warps, **arrays)
        report.write_text(json.dumps({
            "schema": analysis.CMC_ANALYSIS_SCHEMA, "status": "complete", "ground_truth_used": False,
            "build_path": str(args.build_path.resolve()),
            "manifest_sha256": hashlib.sha256((args.build_path / "manifest.json").read_bytes()).hexdigest(),
            "source_signature": analysis.cmc_build_signature(args.build_path),
            "split": args.split, "rule": CMC_ANALYSIS_RULE, **cmc.cmc_tracker_inputs(spec),
            "sequences": sequences, "total_frames": sum(args.seq_info[name] for name in selected),
            "policy": policy.to_dict(), "warps_sha256": hashlib.sha256(warps.read_bytes()).hexdigest(),
        }))
        return SimpleNamespace(policy=policy, report_path=report, warps_path=warps)

    monkeypatch.setattr(cmc, "prepare_cmc_candidates", prepare)
    return calls


@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_automatic_setup_analyzes_current_images_without_ground_truth(auto_inputs, monkeypatch, tmp_path, tracker):
    """The real image analyzer decides new sequences without calibration labels."""
    datasets = {
        name: _Dataset(_sample(name, index) for index in range(count)) for name, count in auto_inputs.seq_info.items()
    }
    for dataset in datasets.values():
        dataset._image_root = str(tmp_path / "images")
        for sample in dataset:
            image = Path(dataset._image_root) / sample.image_ref
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture pixels")
    monkeypatch.setattr(
        analysis.CachedVisionDataset, "_for_sequence", lambda build, *, sequence_id, **kwargs: datasets[sequence_id]
    )
    monkeypatch.setattr(
        analysis, "_bgr_image", lambda sample, root: np.full((100, 100, 3), "moving" in sample.sequence_id, np.uint8)
    )

    class Estimator:
        def __init__(self, **kwargs):
            self.frame = 0

        def apply(self, image, boxes):
            self.was_initializing = self.frame == 0
            self.inliers_accepted = self.transform_valid = self.frame > 0
            self.match_count = self.inlier_count = 30
            warp = np.eye(2, 3, dtype=np.float32)
            if self.frame:
                warp[0, 2] = 3.0 if image[0, 0, 0] else 0.01
            self.frame += 1
            return warp

    monkeypatch.setattr(analysis, "_DiagnosticSOF", Estimator)
    # No annotation path, ground-truth loader, or calibration output is supplied.
    auto_inputs.tracker_class_ids = (1, 2)
    auto_inputs.tracker_class_names = ((1, "person"), (2, "other"))
    auto_inputs.tracker = tracker
    spec = replace(
        _spec(
            **{
                "calibration.cmc_policy": "auto",
                "calibration.cmc_rule": CMC_ANALYSIS_RULE,
                "calibration.cmc_min_conf": 0.2,
                **calibration_cmc_signature(load_tracker_config(tracker)),
            }
        ),
        name=tracker,
        class_ids=(1, 2),
        class_names=((1, "person"), (2, "other")),
    )
    output = tmp_path / "output"
    prepared = cmc.prepare_eval_cmc(auto_inputs, spec, output)
    policy = prepared.policy
    assert dict(policy.sequences) == {"unseen-moving": True, "unseen-static": False}
    assert yaml.safe_load((output / "cmc-policy.yaml").read_text()) == policy.to_dict()
    report = json.loads((output / "cmc" / "analysis.json").read_text())
    assert report["ground_truth_used"] is False
    assert report["tracker"] == tracker
    assert report["mask_min_conf"] == 0.2
    assert set(report["sequences"]) == set(auto_inputs.seq_info)


def test_analysis_finishes_before_tracking_stage(auto_inputs, stub_analysis, monkeypatch, tmp_path):
    """Both setup=False tuning and ordinary eval enter tracking with a policy."""
    output = tmp_path / "output"
    events = []

    def advance(message):
        if message.startswith("Replaying"):
            assert len(stub_analysis) == 1
            assert (output / "cmc-policy.yaml").is_file()
        events.append(message)

    def replay(build, spec, **kwargs):
        assert events[-1].startswith("Replaying")
        assert dict(kwargs["cmc_policy"].sequences)["unseen-moving"]
        return ReplayResult(build, output, (), 8, 0)

    pipeline = SimpleNamespace(advance=advance, update=events.append)
    monkeypatch.setattr(evaluator, "_refresh_eval_pipeline_intro", lambda *args: None)
    monkeypatch.setattr(evaluator, "replay_build", replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", lambda *args, **kwargs: {"HOTA": 60.0})
    evaluator.run_eval(auto_inputs, setup=False, pipeline=pipeline, show_progress=False, output_dir=output)


@pytest.mark.parametrize("policy", [None, "off"])
def test_disabled_policy_never_loads_images(auto_inputs, monkeypatch, tmp_path, policy):
    auto_inputs.cmc_policy = policy

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled camera analysis must not read image inputs.")

    monkeypatch.setattr(cmc, "prepare_cmc_candidates", unexpected)
    result = cmc.prepare_eval_cmc(auto_inputs, _spec(), tmp_path / "output")
    assert result is None if policy is None else result.policy == SequenceCMCPolicy()


def test_selected_subset_alone_is_analyzed(auto_inputs, stub_analysis, tmp_path):
    auto_inputs.sequence_names = ("unseen-static",)
    policy = cmc.prepare_eval_cmc(auto_inputs, _spec(), tmp_path / "output").policy
    assert policy.sequences == (("unseen-static", False),)


def test_session_reuses_analysis_when_only_tracker_or_kf_changes(auto_inputs, stub_analysis, tmp_path):
    with ReplaySession(1) as session:
        first = cmc.prepare_eval_cmc(auto_inputs, _spec(min_conf=0.1), tmp_path / "first", replay_session=session)
        second = cmc.prepare_eval_cmc(
            auto_inputs, _spec(min_conf=0.8, **{"kalman.type": "xywh"}), tmp_path / "second", replay_session=session
        )
    assert len(stub_analysis) == 1
    assert first == second
    for name in ("analysis.json", "warps.npz"):
        assert (tmp_path / "first" / "cmc" / name).read_bytes() == (tmp_path / "second" / "cmc" / name).read_bytes()


@pytest.mark.parametrize(
    "change", ["manifest", "build", "split", "subset", "mask", "runtime-mask", "classes", "per_class", "estimator"]
)
def test_session_cache_key_changes_with_camera_inputs(auto_inputs, stub_analysis, tmp_path, change):
    spec = _spec()
    with ReplaySession(1) as session:
        cmc.prepare_eval_cmc(auto_inputs, spec, tmp_path / "first", replay_session=session)
        if change == "manifest":
            manifest = auto_inputs.build_path / "manifest.json"
            payload = json.loads(manifest.read_text())
            payload["metadata"]["changed"] = True
            manifest.write_text(json.dumps(payload))
        elif change == "build":
            auto_inputs.build_path = tmp_path / "another-build"
            _source_build(auto_inputs.build_path)
        elif change == "split":
            auto_inputs.split = "train"
        elif change == "subset":
            auto_inputs.sequence_names = ("unseen-moving",)
        elif change == "mask":
            spec = _spec(cmc_min_conf=0.5)
        elif change == "runtime-mask":
            spec = _spec(cmc_min_conf=None, min_conf=0.2)
        elif change == "classes":
            spec = replace(spec, class_ids=(2,), class_names=((2, "vehicle"),))
        elif change == "per_class":
            spec = replace(spec, per_class=True)
        elif change == "estimator":
            spec = _spec(sof_target_width=320)
        cmc.prepare_eval_cmc(auto_inputs, spec, tmp_path / "second", replay_session=session)
        assert len(session._cmc_analysis_cache) == 1
    assert len(stub_analysis) == 2


def test_new_eval_reanalyzes_even_when_output_artifacts_exist(auto_inputs, stub_analysis, tmp_path):
    """An existing output directory must never bind new eval to old decisions."""
    output = tmp_path / "output"
    cmc.prepare_eval_cmc(auto_inputs, _spec(), output)
    auto_inputs.seq_info = {"new-static-camera": 3}
    policy = cmc.prepare_eval_cmc(auto_inputs, _spec(), output).policy
    assert len(stub_analysis) == 2
    assert policy.sequences == (("new-static-camera", False),)
    assert yaml.safe_load((output / "cmc-policy.yaml").read_text())["sequences"] == {"new-static-camera": False}


def test_off_clears_only_old_automatic_artifacts(auto_inputs, stub_analysis, tmp_path):
    """Reusing an evaluation output must not leave old automatic evidence active."""
    output = tmp_path / "output"
    cmc.prepare_eval_cmc(auto_inputs, _spec(), output)
    unrelated = output / "cmc" / "notes.txt"
    unrelated.write_text("keep this file")
    auto_inputs.cmc_policy = "off"
    policy = cmc.prepare_eval_cmc(auto_inputs, _spec(), output).policy
    assert policy == SequenceCMCPolicy()
    assert not (output / "cmc" / "analysis.json").exists()
    assert not (output / "cmc" / "warps.npz").exists()
    assert unrelated.read_text() == "keep this file"
    assert len(stub_analysis) == 1


@pytest.mark.parametrize("source", ["image", "artifact"])
def test_changed_source_epoch_reanalyzes_session_without_trusting_restored_mtime(
    auto_inputs, stub_analysis, tmp_path, source
):
    """The same logical settings must not reuse evidence for changed pixels/detections."""
    artifact = auto_inputs.build_path / "instances" / "part-00000.parquet"
    artifact.write_bytes(b"original input")
    with ReplaySession(1) as session:
        prepared = cmc.prepare_eval_cmc(auto_inputs, _spec(), tmp_path / "first", replay_session=session)
        path = artifact if source == "artifact" else Path(prepared.replay_sequences[0].image_signature[0][0])
        original = path.stat()
        path.write_bytes(b"x" * original.st_size)
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        cmc.prepare_eval_cmc(auto_inputs, _spec(), tmp_path / "second", replay_session=session)
    assert len(stub_analysis) == 2


def test_tuning_handoff_reuses_immutable_warps_without_decoding_images(
    auto_inputs, stub_analysis, monkeypatch, tmp_path
):
    prepared = cmc.prepare_tuning_cmc(auto_inputs, _spec(), tmp_path / "driver")
    auto_inputs._cmc_setup = prepared

    def forbidden(*args, **kwargs):
        pytest.fail("A validated tuning handoff must not analyze or decode images.")

    monkeypatch.setattr(cmc, "prepare_cmc_candidates", forbidden)
    monkeypatch.setattr(analysis, "_bgr_image", forbidden)
    actual = cmc.prepare_eval_cmc(auto_inputs, _spec(min_conf=0.8), tmp_path / "trial")
    assert actual == prepared
    assert len(actual.replay_sequences) == 1
    sequence = pickle.loads(pickle.dumps(actual.replay_sequences[0]))
    assert sequence.sequence_id == "unseen-moving"
    assert sequence.sample_ids == tuple(f"unseen-moving/{index}" for index in range(4))
    assert sequence.frame_indices == tuple(range(4))
    assert not sequence.warps.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        sequence.warps[1, 0, 2] = 3
    sequence.validate_sources()
    sequence.validate_selection(auto_inputs.build_path, auto_inputs.split)
    sequence.validate_tracker(prepared.policy.apply(_spec(), sequence.sequence_id))
    with pytest.raises(ValueError, match="build and split"):
        sequence.validate_selection(auto_inputs.build_path, "different-split")
    with pytest.raises(ValueError, match="camera inputs"):
        sequence.validate_tracker(_spec(use_cmc=False))


@pytest.mark.parametrize("change", ["image", "build", "mask", "runtime-mask", "classes", "estimator"])
def test_tuning_handoff_rejects_changed_camera_inputs(auto_inputs, stub_analysis, monkeypatch, tmp_path, change):
    spec = _spec()
    prepared = cmc.prepare_tuning_cmc(auto_inputs, spec, tmp_path / "driver")
    auto_inputs._cmc_setup = prepared
    if change == "image":
        Path(prepared.replay_sequences[0].image_signature[0][0]).write_bytes(b"changed pixels")
    elif change == "build":
        (auto_inputs.build_path / "instances" / "part-00000.parquet").write_bytes(b"changed detections")
    elif change == "mask":
        spec = _spec(cmc_min_conf=0.7)
    elif change == "runtime-mask":
        spec = _spec(cmc_min_conf=None, min_conf=0.2)
    elif change == "classes":
        spec = replace(spec, class_ids=(2,), class_names=((2, "vehicle"),))
    else:
        spec = _spec(sof_target_width=320)
    with pytest.raises(ValueError, match="changed|differ"):
        cmc.prepare_eval_cmc(auto_inputs, spec, tmp_path / "trial")
    assert len(stub_analysis) == 1
    if change in {"image", "build"}:
        with pytest.raises(ValueError, match="source files changed"):
            prepared.replay_sequences[0].validate_sources()


@pytest.mark.parametrize("change", ["order", "count", "nonfinite", "first-frame", "checksum"])
def test_saved_warps_reject_corrupt_frame_alignment(auto_inputs, stub_analysis, tmp_path, change):
    prepared = cmc.prepare_tuning_cmc(auto_inputs, _spec(), tmp_path / "driver")
    report = json.loads(prepared.report)
    with np.load(io.BytesIO(prepared.warps), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if change == "order":
        arrays["unseen-moving_indices"] = arrays["unseen-moving_indices"][::-1]
    elif change == "count":
        arrays["unseen-moving_warps"] = arrays["unseen-moving_warps"][:-1]
    elif change == "nonfinite":
        arrays["unseen-moving_warps"][1, 0, 2] = np.nan
    elif change == "first-frame":
        arrays["unseen-moving_warps"][0, 0, 2] = 1
    else:
        arrays["unseen-moving_warps"][1, 0, 2] = 1
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    encoded = stream.getvalue()
    if change != "checksum":
        report["warps_sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="Saved CMC"):
        cmc._validated_analysis(auto_inputs, _spec(), json.dumps(report).encode(), encoded)
