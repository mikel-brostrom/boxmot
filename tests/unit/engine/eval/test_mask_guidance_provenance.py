"""Prevent guided evaluations from being mistaken for ordinary ByteTrack runs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxmot.engine.eval.evaluator import _output_directory
from boxmot.engine.eval.provenance import mask_guidance_output_path, write_mask_guidance_provenance
from boxmot.trackers import TrackerSpec


def _args(tmp_path: Path, *, checkpoint: Path | None = None, exist_ok: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        project=tmp_path / "runs",
        dataset_id="mot17",
        name="ablation",
        edgetam=checkpoint is not None,
        mask_guidance_weights=checkpoint,
        device="cpu",
        exist_ok=exist_ok,
        tracker="bytetrack",
        geometry="aabb",
        tracker_class_ids=(),
        tracker_class_names=(),
    )


def test_guided_run_never_reuses_ordinary_output_directory_even_with_exist_ok(tmp_path: Path) -> None:
    ordinary = _output_directory(_args(tmp_path), None)
    ordinary_tracks = ordinary / "MOT17-02.txt"
    ordinary_tracks.write_text("ordinary tracking results\n")
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"reference model weights")

    guided = _output_directory(_args(tmp_path, checkpoint=weights), None)

    assert ordinary == tmp_path / "runs/mot17/ablation"
    assert guided.parent == ordinary.parent
    assert guided.name.startswith("ablation-bytetrack-edgetam-")
    assert not (guided / ordinary_tracks.name).exists()
    assert ordinary_tracks.read_text() == "ordinary tracking results\n"
    assert _output_directory(_args(tmp_path), None) == ordinary


def test_pytorch_guidance_outputs_do_not_require_tflite_modules(tmp_path: Path, monkeypatch) -> None:
    """Create OccluBoost/MPS outputs when the optional TFLite code is absent."""
    monkeypatch.setitem(sys.modules, "boxmot.segmentors.exporters.edgetam.bundle", None)
    monkeypatch.setitem(sys.modules, "boxmot.segmentors.propagation.tflite", None)
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"pytorch checkpoint")
    args = _args(tmp_path, checkpoint=weights)
    args.tracker = "occluboost"
    args.device = "mps"
    args.mask_guidance_max_objects = 96

    output = _output_directory(args, None)
    provenance = write_mask_guidance_provenance(
        output,
        checkpoint=weights,
        device=args.device,
        build=tmp_path / "build",
        tracker_spec=TrackerSpec(args.tracker),
        max_objects=args.mask_guidance_max_objects,
    )

    assert output.name.startswith("ablation-occluboost-edgetam-")
    metadata = json.loads(provenance.read_text())
    assert metadata["device"] == "mps"
    assert metadata["precision"] == "fp16"
    assert metadata["propagation"]["max_objects"] == 96
    assert metadata["propagation"]["object_batch_size"] == 4
    assert metadata["matching"]["adjustment"] == "apply_conditioned_cost_delta_to_geometry_and_fused_ranking"


def test_guided_output_identity_tracks_weight_contents_and_device(tmp_path: Path, monkeypatch) -> None:
    from boxmot.segmentors.propagation import model

    monkeypatch.setattr(model, "effective_precision", lambda device: "fp32")
    monkeypatch.setattr(model, "postprocessing_metadata", lambda device: {"fill_hole_area": 0})
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"first model")
    base = tmp_path / "ablation"
    original = mask_guidance_output_path(base, checkpoint=weights, device="cpu")
    copied_weights = tmp_path / "copied.pt"
    copied_weights.write_bytes(weights.read_bytes())

    assert mask_guidance_output_path(base, checkpoint=copied_weights, device="cpu") == original
    assert mask_guidance_output_path(base, checkpoint=weights, device="cuda:0") != original
    assert mask_guidance_output_path(base, checkpoint=weights, device="0") == mask_guidance_output_path(
        base, checkpoint=weights, device="cuda:0"
    )
    weights.write_bytes(b"other model")
    assert mask_guidance_output_path(base, checkpoint=weights, device="cpu") != original


def test_disabled_guidance_ignores_selected_weights_for_output_identity(monkeypatch, tmp_path) -> None:
    """Toggling off restores the ordinary output path without touching model files."""
    from boxmot.segmentors.propagation import weights

    args = _args(tmp_path, checkpoint=tmp_path / "missing.pt")
    args.edgetam = False
    monkeypatch.setattr(weights, "resolve_edgetam_checkpoint", lambda *_args: pytest.fail("Resolved disabled weights"))

    assert _output_directory(args, None) == _output_directory(_args(tmp_path), None)
    assert args.mask_guidance_weights == tmp_path / "missing.pt"


def test_guided_runs_preserve_existing_increment_policy(tmp_path: Path) -> None:
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"model")
    args = _args(tmp_path, checkpoint=weights, exist_ok=False)

    first = _output_directory(args, None)
    second = _output_directory(args, None)

    assert first != second
    assert second.name == f"{first.name}2"


def test_guided_trial_outputs_are_distinct_from_ordinary_trials(tmp_path: Path) -> None:
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"model")
    ordinary = _output_directory(_args(tmp_path), {"track_thresh": 0.5})

    guided = _output_directory(_args(tmp_path, checkpoint=weights), {"track_thresh": 0.5})

    assert guided != ordinary
    assert guided.parent.name == ordinary.parent.name == "trials"
    assert guided.name == ordinary.name
    assert "bytetrack-edgetam" in guided.parent.parent.name


def test_guidance_sidecar_identifies_effective_model_tracker_and_build(tmp_path: Path) -> None:
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"reference model")
    spec = TrackerSpec("bytetrack", options=(("match_thresh", 0.2), ("track_thresh", 0.55)))
    output = tmp_path / "output"
    output.mkdir()
    tracks = output / "MOT17-02.txt"
    tracks.write_text("freshly replayed tracks\n")

    path = write_mask_guidance_provenance(
        output,
        checkpoint=weights,
        device="cpu",
        build=tmp_path / "immutable-build",
        tracker_spec=spec,
        sequence_names=("MOT17-02",),
    )
    saved = json.loads(path.read_text())

    assert saved["method"] == "bytetrack-edgetam"
    assert saved["checkpoint_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert saved["checkpoint"] == str(weights.resolve())
    assert saved["device"] == "cpu"
    assert saved["reference"]["commit"] == "7711e012a30a2402c4eaab637bdb00a521302c91"
    assert saved["schema"] == "boxmot.mask-guidance-evaluation/v4"
    assert saved["policy_version"] == 6
    assert saved["precision"] == "fp32"
    assert saved["propagation"]["max_objects"] == 96
    assert saved["propagation"]["object_batch_size"] == 4
    assert saved["propagation"]["prompt_overlap"] == 0.10
    assert saved["build"] == str((tmp_path / "immutable-build").resolve())
    assert saved["sequence_names"] == ["MOT17-02"]
    assert saved["tracker"]["name"] == "bytetrack"
    assert saved["resolved_tracker_options"]["track_thresh"] == 0.55
    assert saved["resolved_tracker_options"]["match_thresh"] == 0.2
    assert saved["matching"] == {
        "min_mask_coverage": 0.90,
        "min_mask_fill": 0.05,
        "candidate_matrix": "stage_cost",
        "high_threshold": 0.2,
        "low_threshold": 0.5,
        "unconfirmed_threshold": 0.7,
        "candidate_policy": "original_cost_at_or_below_gate_and_ambiguous",
        "isolation_recovery": "disabled",
        "clear_match_policy": "add_10_to_competing_pairs_per_clear_row_and_column",
        "rasterization": "truncate_tlwh_clamp_origin_then_clip_bounds",
    }
    assert tracks.read_text() == "freshly replayed tracks\n"
    assert sorted(item.name for item in output.iterdir()) == ["MOT17-02.txt", "mask-guidance.json"]


def test_missing_guidance_checkpoint_does_not_allocate_an_output_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="EdgeTAM checkpoint does not exist"):
        _output_directory(_args(tmp_path, checkpoint=tmp_path / "missing.pt"), None)

    assert not (tmp_path / "runs").exists()


def test_guidance_identity_includes_budget_and_actual_matching_threshold(tmp_path: Path) -> None:
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"model")
    base = tmp_path / "output"
    default = mask_guidance_output_path(base, checkpoint=weights, device="cpu")
    capped = mask_guidance_output_path(base, checkpoint=weights, device="cpu", max_objects=4)
    tuned = mask_guidance_output_path(
        base,
        checkpoint=weights,
        device="cpu",
        tracker_spec=TrackerSpec("bytetrack", options=(("match_thresh", 0.2),)),
    )
    assert len({default, capped, tuned}) == 3


def test_changed_object_batching_changes_guided_output_identity(tmp_path: Path, monkeypatch) -> None:
    from boxmot.segmentors.propagation import edgetam

    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"checkpoint")
    base = tmp_path / "output"
    default = mask_guidance_output_path(base, checkpoint=weights, device="cpu")

    monkeypatch.setattr(edgetam, "DEFAULT_OBJECT_BATCH_SIZE", 1)

    assert mask_guidance_output_path(base, checkpoint=weights, device="cpu") != default


@pytest.mark.parametrize(
    ("parameter", "value", "section", "field"),
    [
        ("edgetam.min_coverage", 0.75, "matching", "min_mask_coverage"),
        ("edgetam.min_fill", 0.12, "matching", "min_mask_fill"),
        ("edgetam.prompt_overlap", 0.25, "propagation", "prompt_overlap"),
        ("edgetam.max_objects", 8, "propagation", "max_objects"),
    ],
)
def test_each_tuned_mask_setting_changes_fingerprint_and_effective_provenance(
    tmp_path: Path, parameter: str, value: float | int, section: str, field: str
) -> None:
    """Two trials with different mask behavior cannot publish the same identity."""
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"checkpoint")
    base = tmp_path / "eval"
    spec = TrackerSpec("bytetrack", options=((parameter, value),))
    ordinary_settings = mask_guidance_output_path(base, checkpoint=weights, device="cpu")
    tuned_settings = mask_guidance_output_path(base, checkpoint=weights, device="cpu", tracker_spec=spec)

    assert tuned_settings != ordinary_settings
    path = write_mask_guidance_provenance(
        tuned_settings, checkpoint=weights, device="cpu", build=tmp_path / "build", tracker_spec=spec
    )
    saved = json.loads(path.read_text())
    assert saved[section][field] == value
    assert saved["resolved_tracker_options"][parameter] == value


def test_cap_override_and_tracker_option_have_the_same_effective_fingerprint(tmp_path: Path) -> None:
    weights = tmp_path / "edgetam.pt"
    weights.write_bytes(b"checkpoint")
    assert mask_guidance_output_path(tmp_path / "eval", checkpoint=weights, device="cpu", max_objects=8) == (
        mask_guidance_output_path(
            tmp_path / "eval",
            checkpoint=weights,
            device="cpu",
            tracker_spec=TrackerSpec("bytetrack", options=(("edgetam.max_objects", 8),)),
        )
    )


@pytest.mark.parametrize(
    ("name", "threshold"),
    [
        ("bytetrack", "match_thresh"),
        ("botsort", "second_match_thresh"),
        ("strongsort", "max_cos_dist"),
        ("sfsort", "match_th_first_m"),
        ("ocsort", "iou_threshold"),
        ("deepocsort", "w_association_emb"),
        ("hybridsort", "longterm_reid_correction_thresh"),
        ("boosttrack", "lambda_iou"),
        ("occluboost", "recovery_appearance_thresh"),
    ],
)
def test_each_tracker_records_its_policy_and_fingerprints_its_options(
    tmp_path: Path, name: str, threshold: str
) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"checkpoint")
    spec = TrackerSpec(name, options=(("asso_func", "iou"),))
    tuned = TrackerSpec(name, options=tuple(sorted((("asso_func", "iou"), (threshold, 0.123456)))))
    original_path = mask_guidance_output_path(tmp_path / "eval", checkpoint=weights, device="cpu", tracker_spec=spec)
    changed_path = mask_guidance_output_path(tmp_path / "eval", checkpoint=weights, device="cpu", tracker_spec=tuned)
    assert original_path != changed_path
    assert changed_path.name.startswith(f"eval-{name}-edgetam-")
    path = write_mask_guidance_provenance(
        changed_path, checkpoint=weights, device="cpu", build=tmp_path / "build", tracker_spec=tuned
    )
    saved = json.loads(path.read_text())
    assert saved["method"] == f"{name}-edgetam"
    assert saved["resolved_tracker_options"][threshold] == 0.123456
    assert saved["matching"]["candidate_matrix"]


@pytest.mark.parametrize(("high", "low", "expected"), [(None, None, (0.67, 0.3)), (1.5, -0.5, (0.67, 0.0))])
def test_sfsort_provenance_reports_normalized_gates(
    tmp_path: Path, high: float | None, low: float | None, expected: tuple[float, float]
) -> None:
    from boxmot import SFSORT, SFSORTConfig

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"checkpoint")
    spec = TrackerSpec(
        "sfsort",
        options=(
            ("dynamic_tuning", True),
            ("match_th_first", high),
            ("match_th_first_m", None),
            ("match_th_second", low),
        ),
    )
    path = write_mask_guidance_provenance(
        tmp_path / "eval", checkpoint=weights, device="cpu", build=tmp_path / "build", tracker_spec=spec
    )
    saved = json.loads(path.read_text())
    policy = saved["matching"]
    assert (policy["high_threshold"], policy["low_threshold"]) == expected
    assert policy["dynamic_threshold"]["enabled"]
    assert policy["dynamic_threshold"]["multiplier"] == 0.02
    assert saved["resolved_tracker_options"]["match_th_first"] == high
    tracker = SFSORT(config=SFSORTConfig.from_mapping(spec.option_dict))
    assert policy["high_threshold"] == tracker.match_th_first
    assert policy["low_threshold"] == tracker.match_th_second
    assert policy["dynamic_threshold"]["multiplier"] == tracker.match_th_first_m
    assert policy["dynamic_threshold"]["confidence_cutoff"] == tracker.cth
