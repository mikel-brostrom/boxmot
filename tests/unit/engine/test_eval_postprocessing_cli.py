"""Evaluation postprocessing is ordered and requests the artifacts it consumes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.postprocessing import normalize_postprocessing
from boxmot.engine.config.runtime import BOXMOT_DEFAULTS, build_mode_namespace, get_mode_defaults
from tests._paths import REPO_ROOT


@pytest.mark.parametrize("methods", [(), ("gsi",), ("gbrc",), ("gta",), ("gta", "gbrc", "gsi")])
def test_eval_forwards_requested_postprocessing_order(
    methods: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: captured.update(args=vars(args))),
    )
    flags = [flag for method in methods for flag in ("--postprocessing", method)]
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "existing-build", *flags],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"]["postprocessing"] == methods


@pytest.mark.parametrize("tracker", ["bytetrack", "botsort"])
@pytest.mark.parametrize("method", ["gta", "gsi", "gbrc"])
@pytest.mark.parametrize("component_selectors", [False, True])
def test_materialization_publishes_only_embeddings_required_by_postprocessing(
    tracker: str, method: str, component_selectors: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def materialize(args):
        captured["materialize"] = args
        return tmp_path / "build"

    monkeypatch.setitem(sys.modules, "boxmot.engine.materialization.workflow", SimpleNamespace(main=materialize))
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: captured.update(eval=args)),
    )
    tracker_flags = []
    if tracker == "botsort":
        profile = tmp_path / "motion.yaml"
        profile.write_text("use_embeddings: false\n", encoding="utf-8")
        tracker_flags = ["--tracker-config", str(profile)]
    input_flags = (
        ["--dataset", "mot17", "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke", "--split", "ablation"]
        if component_selectors
        else ["--experiment", "mot17/ablation-yolox-lmbn.yaml"]
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            *input_flags,
            "--tracker",
            tracker,
            "--postprocessing",
            method,
            *tracker_flags,
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["materialize"].publish_embeddings is (method == "gta")
    assert captured["eval"].postprocessing == (method,)
    assert captured["eval"].build == tmp_path / "build"


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--postprocessing", "gsi", "--postprocessing", "gsi"], "only once"),
        (["--postprocessing", "gbrc", "--postprocessing", "gta"], "GTA must run before"),
        (["--postprocessing", "gsi", "--postprocessing", "gta"], "GTA must run before"),
        (["--postprocessing", "invalid"], "not one of"),
        (["--postprocessing", "gsi", "--eval-masks"], "image AABB evaluation only"),
        (["--postprocessing", "gsi", "--eval-3d"], "image AABB evaluation only"),
    ],
)
def test_invalid_postprocessing_fails_before_workflow(
    flags: list[str], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(_args):
        pytest.fail("Invalid postprocessing must fail before materialization or replay.")

    for module in ("boxmot.engine.materialization.workflow", "boxmot.engine.eval.evaluator"):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=unexpected))
    result = CliRunner().invoke(boxmot, ["eval", "--experiment", "mot17/ablation-yolox-lmbn.yaml", *flags])
    assert result.exit_code == 2, (result.output, result.exception)
    assert message in result.output


def test_postprocessing_rejects_obb_before_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda _args: pytest.fail("Unsupported experiment materialized")),
    )
    result = CliRunner().invoke(
        boxmot, ["eval", "--experiment", "mmot-obb/test-yolo11l-lmbn.yaml", "--postprocessing", "gsi"]
    )
    assert result.exit_code == 2, (result.output, result.exception)
    assert "requires an image AABB dataset" in result.output


@pytest.mark.parametrize("detector_embeddings", (False, True))
def test_gta_materializer_resolves_embedding_availability_without_reid(
    detector_embeddings: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only materialization has the resolved detector capability needed to decide."""
    from boxmot.detectors import DetectorCapabilities, DetectorSpec
    from boxmot.engine.materialization import workflow
    from boxmot.engine.materialization.catalog import SourceCatalog
    from boxmot.engine.materialization.source import SourceSample

    experiment = yaml.safe_load((REPO_ROOT / "boxmot/configs/experiments/mot17/ablation-yolox-lmbn.yaml").read_text())
    experiment.pop("reid")
    experiment_path = tmp_path / "no-reid.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment))

    sample = SourceSample(
        sample_id="val:sequence:0",
        split="val",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=(tmp_path / "frame.jpg").as_uri(),
        source_sha256="a" * 64,
    )
    catalog = SourceCatalog(samples=(sample,), fingerprint="b" * 64, source_root=tmp_path, metadata={})
    detector = DetectorSpec("fixture", geometry_mode="aabb", options=(("embedding_dim", 4),))
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda _args, **_kwargs: ("fixture", "aabb", catalog, "fixture-detector", None, None, {}),
    )
    monkeypatch.setattr(
        workflow, "resolve_detector_spec", lambda *_args, **_kwargs: (detector, {"spec": {"backend": "fixture"}})
    )
    monkeypatch.setattr(
        workflow, "detector_capabilities", lambda _spec: DetectorCapabilities(provides_embeddings=detector_embeddings)
    )
    monkeypatch.setattr(
        workflow, "resolve_reid_spec", lambda *_args, **_kwargs: pytest.fail("An unauthored ReID model was requested")
    )
    monkeypatch.setattr(workflow, "import_former_default_build", lambda *_args, **_kwargs: None)
    captured = {}

    def reuse_build(plan, **_kwargs):
        captured["plan"] = plan
        return plan.output_root

    monkeypatch.setattr(workflow, "find_matching_build", reuse_build)
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: captured.update(eval=args)),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            str(experiment_path),
            "--build-root",
            str(tmp_path / "builds"),
            "--postprocessing",
            "gta",
        ],
    )
    if detector_embeddings:
        assert result.exit_code == 0, (result.output, result.exception)
        assert captured["plan"].publish.embeddings is True
        assert [stage.name for stage in captured["plan"].stages] == ["detect", "finalize"]
        assert captured["eval"].build == captured["plan"].output_root
        assert captured["eval"].postprocessing == ("gta",)
    else:
        assert result.exit_code == 1, (result.output, result.exception)
        assert isinstance(result.exception, ValueError)
        assert "Published embeddings require the selected experiment to define ReID" in str(result.exception)
        assert "its detector does not provide embeddings" in str(result.exception)
        assert "eval" not in captured


@pytest.mark.parametrize("method", ["gsi", "gbrc", "gta"])
@pytest.mark.parametrize("saved_boxes", [False, True])
def test_sensor_and_saved_predictions_reject_postprocessing_before_loading_payloads(
    method: str, saved_boxes: bool, tmp_path: Path
) -> None:
    source = REPO_ROOT / "boxmot/configs/datasets/sensor-fusion.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if saved_boxes:
        config["modalities"] = {
            key: value for key, value in config["modalities"].items() if key in {"images", "detections_2d"}
        }
        config["modalities"]["detections_2d"]["options"] = {"load_masks": False}
        config["modalities"]["ground_truth"] = {"format": "kitti-tracking-labels", "path": "labels/{sequence}.txt"}
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            str(dataset),
            "--tracker",
            "bytetrack" if saved_boxes else "eagermot",
            "--postprocessing",
            method,
        ],
    )
    assert result.exit_code == 2, (result.output, result.exception)
    assert "does not support --postprocessing" in result.output


def test_python_eval_normalizes_postprocessing_and_defaults() -> None:
    assert BOXMOT_DEFAULTS.eval.postprocessing == ()
    assert get_mode_defaults("eval")["postprocessing"] == ()
    assert build_mode_namespace("eval", {}).postprocessing == ()
    assert build_mode_namespace("eval", {"postprocessing": "gsi"}).postprocessing == ("gsi",)
    assert build_mode_namespace("eval", {"postprocessing": ["gta", "gbrc"]}).postprocessing == ("gta", "gbrc")


@pytest.mark.parametrize("value", [["gsi", "gta"], ["gsi", "gsi"], "GTA", 1, {"gta"}, [None]])
def test_python_eval_rejects_ambiguous_postprocessing(value: object) -> None:
    with pytest.raises(ValueError):
        build_mode_namespace("eval", {"postprocessing": value})
    with pytest.raises(ValueError):
        normalize_postprocessing(value)
