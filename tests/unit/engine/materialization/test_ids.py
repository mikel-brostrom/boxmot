from __future__ import annotations

from dataclasses import replace

import pytest

from boxmot.datasets.manifest import StageProvenance
from boxmot.engine.materialization.ids import component_content, fingerprint, make_stage_fingerprint, stage_content


def _stages(device: str, **detector_spec: str) -> tuple[StageProvenance, ...]:
    """Construct a real fingerprint chain with two parallel detector consumers."""

    stages: dict[str, StageProvenance] = {}
    for name, inputs in (
        ("detect", ()),
        ("segment", ("detect",)),
        ("embed", ("detect",)),
        ("finalize", ("segment", "embed")),
    ):
        component = (
            {}
            if name == "finalize"
            else {
                "spec": {
                    "device": device,
                    "precision": "fp32",
                    "preprocessing": "default",
                    "artifact_sha256": "a" * 64,
                    **(detector_spec if name == "detect" else {}),
                }
            }
        )
        stages[name] = StageProvenance(
            name=name,
            fingerprint=make_stage_fingerprint(
                name,
                config={},
                component=component,
                upstream=tuple(stages[dependency].fingerprint for dependency in inputs),
            ),
            component=component,
            inputs=inputs,
            batch_size=2,
        )
    return tuple(stages.values())


def test_component_content_removes_only_spec_device_without_mutating_provenance() -> None:
    provenance = {
        "device": "top-level-semantic-value",
        "spec": {
            "device": "cuda:0",
            "precision": "fp16",
            "options": {"device": "backend-option"},
        },
    }

    assert component_content(provenance) == {
        "device": "top-level-semantic-value",
        "spec": {"precision": "fp16", "options": {"device": "backend-option"}},
    }
    assert provenance["spec"]["device"] == "cuda:0"


@pytest.mark.parametrize("device", ("cpu", "mps", "cuda:0"))
def test_stage_content_normalizes_device_and_downstream_fingerprints(device: str) -> None:
    stages = _stages(device)
    normalized = stage_content(stages)

    assert normalized == stage_content(_stages("cpu"))
    assert stage_content(normalized) == normalized
    assert all(
        original.fingerprint == content.fingerprint for original, content in zip(stages, normalized, strict=True)
    )
    assert all(stage.component["spec"]["device"] == device for stage in stages[:-1])


def test_stage_content_resolves_dependencies_before_supplied_order() -> None:
    original = _stages("mps")

    assert stage_content(tuple(reversed(original))) == tuple(reversed(stage_content(original)))


@pytest.mark.parametrize("stage_index", (0, 1, 2, 3))
def test_stage_content_preserves_unverified_fingerprint_drift(stage_index: int) -> None:
    stages = list(_stages("mps"))
    stages[stage_index] = replace(stages[stage_index], fingerprint="f" * 64)

    normalized = stage_content(tuple(stages))

    assert normalized[stage_index].fingerprint == "f" * 64
    assert normalized != stage_content(_stages("cpu"))


@pytest.mark.parametrize(
    "detector_spec",
    ({"precision": "fp16"}, {"preprocessing": "alternate"}, {"artifact_sha256": "b" * 64}),
)
def test_stage_content_preserves_inference_contract_changes(detector_spec: dict[str, str]) -> None:
    assert stage_content(_stages("mps", **detector_spec)) != stage_content(_stages("cpu"))


def test_stage_content_rejects_missing_dependencies() -> None:
    with pytest.raises(ValueError, match="unknown dependencies"):
        stage_content((_stages("cpu")[-1],))


def test_stage_content_rejects_cycles() -> None:
    detect, segment, *_ = _stages("cpu")
    with pytest.raises(ValueError, match="contains a cycle"):
        stage_content((replace(detect, inputs=("segment",)), segment))


def test_recorded_execution_hashes_normalize_to_current_content() -> None:
    """Already published batch-2/device-specific chains remain reusable."""
    recorded = {}
    for stage in _stages("cuda:0"):
        digest = fingerprint({
            "name": stage.name, "config": stage.config, "component": stage.component,
            "upstream": [recorded[name].fingerprint for name in stage.inputs], "batch_size": stage.batch_size,
        })
        recorded[stage.name] = replace(stage, fingerprint=digest)
    assert stage_content(tuple(recorded.values())) == stage_content(_stages("cpu"))


@pytest.mark.parametrize(
    "options", ({"batch_size": 2, "embedding_dim": 512}, (("batch_size", 2), ("embedding_dim", 512)))
)
def test_component_batching_is_execution_only(options) -> None:
    first = {"spec": {"device": "cpu", "options": options}}
    changed = {"spec": {"device": "cuda:0", "options": (
        {"batch_size": 16, "embedding_dim": 512} if isinstance(options, dict)
        else (("batch_size", 16), ("embedding_dim", 512))
    )}}
    assert component_content(first) == component_content(changed)
    assert "embedding_dim" in str(component_content(first))
