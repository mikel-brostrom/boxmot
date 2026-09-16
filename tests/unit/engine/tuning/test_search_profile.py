"""Resume must not restore optimizer dimensions removed by current filtering."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from boxmot.engine.tuning.search_profile import record_search_profile


def _args(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "tracker": "occluboost",
            "tracker_backend": "python",
            "geometry": "aabb",
            "search_alg": "optuna",
            "resume_tune": None,
            **overrides,
        }
    )


def _schema() -> dict:
    return {
        "max_age": {"type": "randint", "default": 146, "range": (15, 200)},
        "use_cmc": {
            "type": "choice",
            "default": True,
            "options": (False, True),
            "activates": {"cmc_method": {"type": "choice", "default": "sof", "options": ("sof", "ecc")}},
        },
        "asso_func": {"default": "iou"},
    }


def test_new_search_records_detached_canonical_schema_and_fixed_values(tmp_path: Path) -> None:
    args, schema = _args(), _schema()
    fixed = {"asso_func": "iou", "kalman.noise.process_position_scale": 0.25}
    schema_before, fixed_before, args_before = deepcopy(schema), deepcopy(fixed), vars(args).copy()
    directory = tmp_path / "new-run"

    record_search_profile(directory, args, MappingProxyType(schema), MappingProxyType(fixed))

    path = directory / "search-space.json"
    content = path.read_bytes()
    saved = json.loads(content)
    assert saved == {
        "version": 1,
        "tracker": "occluboost",
        "tracker_backend": "python",
        "geometry": "aabb",
        "search_alg": "optuna",
        "schema": json.loads(json.dumps(schema)),
        "fixed_options": fixed,
    }
    assert content == (json.dumps(saved, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert schema == schema_before and fixed == fixed_before and vars(args) == args_before
    schema["max_age"]["range"] = (1, 2)
    fixed["asso_func"] = "giou"
    assert path.read_bytes() == content


def test_resume_accepts_equivalent_json_containers_and_mapping_order_without_rewrite(tmp_path: Path) -> None:
    schema = _schema()
    fixed = {"kalman.noise.process_position_scale": 0.25, "asso_func": "iou"}
    record_search_profile(tmp_path, _args(), schema, fixed)
    path = tmp_path / "search-space.json"
    original, modified = path.read_bytes(), path.stat().st_mtime_ns

    record_search_profile(
        tmp_path,
        _args(resume_tune=str(tmp_path)),
        dict(reversed(list(json.loads(json.dumps(schema)).items()))),
        dict(reversed(list(fixed.items()))),
    )

    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == modified


@pytest.mark.parametrize(
    "changed",
    ["tracker", "tracker_backend", "geometry", "search_alg", "schema", "condition", "fixed_options", "fixed_type"],
)
def test_changed_search_profile_rejects_resume_without_overwriting_metadata(tmp_path: Path, changed: str) -> None:
    args, schema, fixed = _args(), _schema(), {"kalman.noise.process_position_scale": 0.25, "flag": True}
    record_search_profile(tmp_path, args, schema, fixed)
    path = tmp_path / "search-space.json"
    original = path.read_bytes()
    args.resume_tune = str(tmp_path)
    if changed in {"tracker", "tracker_backend", "geometry", "search_alg"}:
        setattr(
            args,
            changed,
            {"tracker": "botsort", "tracker_backend": "cpp", "geometry": "obb", "search_alg": "random"}[changed],
        )
    elif changed == "schema":
        schema["obb_max_age"] = {"type": "randint", "default": 30, "range": [1, 200]}
    elif changed == "condition":
        schema.update(schema["use_cmc"].pop("activates"))
    elif changed == "fixed_type":
        fixed["flag"] = 1  # Equality alone would incorrectly equate this to True.
    else:
        fixed["kalman.noise.process_position_scale"] = 0.50

    with pytest.raises(ValueError, match="does not match.*start a new tuning run"):
        record_search_profile(tmp_path, args, schema, fixed)

    assert path.read_bytes() == original


def test_resume_without_profile_rejects_old_run_and_creates_nothing(tmp_path: Path) -> None:
    directory = tmp_path / "old-run"

    with pytest.raises(ValueError, match="predates search-space metadata.*start a new tuning run"):
        record_search_profile(directory, _args(resume_tune=str(directory)), _schema(), {})

    assert not directory.exists()


@pytest.mark.parametrize("payload", [b"{broken", b"\xff", b'{"invalid":NaN}'])
def test_invalid_saved_profile_rejects_resume_without_repair(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "search-space.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="metadata is invalid.*start a new tuning run"):
        record_search_profile(tmp_path, _args(resume_tune=str(tmp_path)), _schema(), {})

    assert path.read_bytes() == payload


def test_existing_profile_is_not_overwritten_even_without_resume_flag(tmp_path: Path) -> None:
    record_search_profile(tmp_path, _args(), _schema(), {})
    path = tmp_path / "search-space.json"
    original = path.read_bytes()

    with pytest.raises(ValueError, match="does not match"):
        record_search_profile(tmp_path, _args(geometry="obb"), _schema(), {})

    assert path.read_bytes() == original
