"""Shipped fusion profiles search only controls used by the selected algorithm."""

from __future__ import annotations

from pathlib import Path

import pytest

from boxmot.engine.tuning.conditions import map_entries
from boxmot.engine.tuning.postprocessing import write_trial_yaml
from boxmot.engine.tuning.search_space import (
    condition_tracker_schema,
    default_tune_config,
    expand_yaml_groups,
    flatten_yaml_config,
    load_yaml_config,
)
from boxmot.trackers.common.config import get_tracker_config_class, load_tracker_config, load_tracker_defaults
from tests.unit.configs.test_centroid_tuning import _samples

IMAGE_TRACKERS = (
    "botsort", "bytetrack", "boosttrack", "occluboost", "ocsort",
    "deepocsort", "hybridsort", "strongsort", "sfsort", "maf_hda",
)
MOTION_TRACKERS = set(IMAGE_TRACKERS) - {"sfsort", "maf_hda"}
APPEARANCE_TRACKERS = {"botsort", "boosttrack", "occluboost", "deepocsort", "hybridsort", "strongsort"}
FUSION_KEYS = {
    "motion_fusion", "motion_fusion_lambda", "motion_fusion_only_position",
    "mc_lambda", "gating_only_position", "fuse_iou", "fuse_detection_score",
    "fuse_first_associate", "fuse_unconfirmed_associate",
}


def _motion_children(name: str) -> tuple[str, str]:
    """StrongSORT retains its existing canonical names for motion settings."""
    return (
        ("mc_lambda", "gating_only_position") if name == "strongsort"
        else ("motion_fusion_lambda", "motion_fusion_only_position")
    )


def _fusion_schema(name: str, **fixed: object) -> tuple[dict, dict]:
    """Keep authored fusion branches while fixing unrelated tuning dimensions."""
    defaults = {**load_tracker_defaults(name), **fixed}

    def freeze(key: str, entry: dict) -> dict:
        if key not in fixed and key in FUSION_KEYS | {"use_embeddings"}:
            return entry
        return {"default": defaults[key], **({"activates": entry["activates"]} if "activates" in entry else {})}

    schema = map_entries(expand_yaml_groups(load_yaml_config(name)), freeze)
    return condition_tracker_schema(schema, defaults, geometry="aabb"), defaults


@pytest.mark.parametrize("name", (*IMAGE_TRACKERS, "eagermot"))
def test_shipped_fusion_controls_match_tracker_capabilities(name: str) -> None:
    """Do not expose ineffective appearance or Kalman controls to other algorithms."""
    expected: set[str] = set()
    if name in IMAGE_TRACKERS:
        expected |= (
            {"fuse_first_associate", "fuse_unconfirmed_associate"}
            if name in {"bytetrack", "botsort"} else {"fuse_detection_score"}
        )
    if name in MOTION_TRACKERS:
        expected |= {"motion_fusion", *_motion_children(name)}
    if name in APPEARANCE_TRACKERS:
        expected.add("fuse_iou")
    assert set(flatten_yaml_config(load_yaml_config(name))) & FUSION_KEYS == expected
    assert set(get_tracker_config_class(name).fields()) & FUSION_KEYS == expected


@pytest.mark.parametrize("name", IMAGE_TRACKERS)
def test_fusion_baseline_roundtrip_preserves_existing_defaults(name: str, tmp_path: Path) -> None:
    """An exported baseline remains a valid runtime profile with the original behavior."""
    defaults = load_tracker_defaults(name)
    schema = load_yaml_config(name)
    baseline = default_tune_config(schema, defaults=defaults)
    path = tmp_path / f"{name}.yaml"
    write_trial_yaml(schema, baseline, path, base_config=defaults, tracker_name=name)
    restored = load_tracker_config(name, path)
    config_type = get_tracker_config_class(name)
    config = config_type.from_mapping({key: restored[key] for key in config_type.fields()})
    assert config == config_type()
    if name in MOTION_TRACKERS:
        assert config.motion_fusion == ("blend" if name == "strongsort" else "off")
        weight, position = _motion_children(name)
        assert getattr(config, weight) == 0.98
        assert getattr(config, position) is False
    if name in APPEARANCE_TRACKERS:
        assert config.fuse_iou is False
    if name in {"bytetrack", "botsort"}:
        assert config.fuse_first_associate is config.fuse_unconfirmed_associate is True
    else:
        assert config.fuse_detection_score is False


@pytest.mark.parametrize("backend", ("random", "optuna", "hyperopt"))
@pytest.mark.parametrize("name", IMAGE_TRACKERS)
def test_fixed_fusion_modes_exclude_inactive_optimizer_dimensions(name: str, backend: str) -> None:
    """Every real backend omits disabled motion/appearance distributions entirely."""
    runtime_defaults = load_tracker_defaults(name)
    modes = ("off", "gate", "blend") if name in MOTION_TRACKERS else (None,)
    for mode in modes:
        fixed = {"motion_fusion": mode} if mode is not None else {}
        if "use_embeddings" in runtime_defaults:
            fixed["use_embeddings"] = False
        schema, defaults = _fusion_schema(name, **fixed)
        flat = flatten_yaml_config(schema)
        sampled = next(_samples(schema, backend))
        if name in MOTION_TRACKERS:
            weight, position = _motion_children(name)
            for key, enabled in ((weight, mode == "blend"), (position, mode != "off")):
                assert (key in sampled) is enabled, (name, mode, key, backend)
                if not enabled:
                    assert flat[key] == {"default": defaults[key]}
        if name in APPEARANCE_TRACKERS and "use_embeddings" in defaults:
            assert "fuse_iou" not in sampled
            assert flat["fuse_iou"] == {"default": False}
        resolved = {**defaults, **sampled}
        config_type = get_tracker_config_class(name)
        config_type.from_mapping({key: resolved[key] for key in config_type.fields()})


@pytest.mark.parametrize("backend", ("random", "optuna", "hyperopt"))
@pytest.mark.parametrize("name", IMAGE_TRACKERS)
def test_sampled_fusion_branches_preserve_inactive_values(name: str, backend: str) -> None:
    """Conditional children activate correctly when their selectors are themselves sampled."""
    schema, defaults = _fusion_schema(name)
    config_type = get_tracker_config_class(name)
    motion_modes = set()
    appearance_modes = set()
    enabled_iou_values = set()
    for sampled in _samples(schema, backend, count=32):
        resolved = {**defaults, **sampled}
        config = config_type.from_mapping({key: resolved[key] for key in config_type.fields()})
        if name in MOTION_TRACKERS:
            mode = config.motion_fusion
            motion_modes.add(mode)
            weight, position = _motion_children(name)
            if mode == "blend":
                assert 0.0 <= getattr(config, weight) <= 1.0
            else:
                assert getattr(config, weight) == defaults[weight]
            if mode == "off":
                assert getattr(config, position) is False
        if name in APPEARANCE_TRACKERS:
            embeddings = resolved.get("use_embeddings", True)
            appearance_modes.add(embeddings)
            if embeddings:
                enabled_iou_values.add(config.fuse_iou)
            else:
                assert config.fuse_iou is False
    if name in MOTION_TRACKERS:
        assert motion_modes == {"off", "gate", "blend"}
    if name in APPEARANCE_TRACKERS:
        assert appearance_modes == ({True, False} if "use_embeddings" in defaults else {True})
        assert enabled_iou_values == {True, False}
