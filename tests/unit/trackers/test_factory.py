"""Name-based tracker construction shares canonical specification semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boxmot import create_tracker
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.common import factory
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST
from boxmot.trackers.common.specs import TrackerSpec
from tests.unit.trackers._reid import RecordingEncoder


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_registered_names_construct_the_same_python_tracker_as_specs(name: str) -> None:
    """Every public algorithm supports shorthand without requiring model loading."""
    shorthand = create_tracker(name)
    explicit = create_tracker(TrackerSpec(name))

    assert type(shorthand) is type(explicit)
    assert type(shorthand).__name__ == _TRACKER_MANIFEST[name].class_path.rsplit(".", 1)[-1]
    assert shorthand.requirements == explicit.requirements
    assert shorthand.capabilities == explicit.capabilities
    assert shorthand.is_obb == explicit.is_obb
    assert shorthand.per_class == explicit.per_class
    assert shorthand.max_age == explicit.max_age


def test_algorithm_keywords_override_mapping_and_spec_without_mutating_inputs() -> None:
    original = TrackerSpec(
        "occluboost",
        per_class=True,
        options=(("max_age", 8), ("min_hits", 2), ("use_embeddings", True)),
    )
    mapping = {"max_age": 19, "min_hits": 4, "use_cmc": False}

    tracker = create_tracker(original, options=mapping, max_age=30, use_embeddings=False)

    assert tracker.max_age == 30
    assert tracker.min_hits == 4
    assert tracker.per_class is True
    assert tracker.use_cmc is False
    assert tracker.requirements.embeddings is False
    assert original == TrackerSpec(
        "occluboost", per_class=True, options=(("max_age", 8), ("min_hits", 2), ("use_embeddings", True))
    )
    assert mapping == {"max_age": 19, "min_hits": 4, "use_cmc": False}


def test_spec_keyword_argument_and_omitted_options_keep_declared_defaults() -> None:
    original = TrackerSpec("occluboost", options=(("min_hits", 2),))
    defaults = load_tracker_defaults("occluboost")

    tracker = create_tracker(spec=original, max_age=30)

    assert tracker.max_age == 30
    assert tracker.min_hits == 2
    assert tracker.det_thresh == defaults["det_thresh"]
    assert tracker.use_embeddings == defaults["use_embeddings"]
    assert original.option_dict == {"min_hits": 2}


def test_selection_keywords_override_spec_and_normalize_class_metadata() -> None:
    original = TrackerSpec("occluboost", geometry="obb", per_class=True, class_ids=(1,))
    ids = [2, 0, 2]
    names = {2: "car", 0: "person"}

    tracker = create_tracker(
        original,
        geometry="aabb",
        per_class=False,
        class_ids=ids,
        class_names=names,
        use_embeddings=True,
    )

    assert tracker.is_obb is False
    assert tracker.per_class is False
    assert tracker.class_ids == frozenset({0, 2})
    assert tracker.class_names == names
    assert tracker.requirements.embeddings is True
    assert original.geometry == "obb" and original.per_class is True
    assert original.class_ids == (1,)
    assert ids == [2, 0, 2] and names == {2: "car", 0: "person"}


def test_class_metadata_can_be_cleared_explicitly() -> None:
    spec = TrackerSpec("bytetrack", class_ids=(0,), class_names=((0, "person"),))

    tracker = create_tracker(spec, class_ids=None, class_names={})

    assert tracker.class_ids is None
    assert tracker.class_names == {}


@pytest.mark.parametrize("name", [name for name, entry in _TRACKER_MANIFEST.items() if entry.native_class_path])
def test_name_and_spec_native_dispatch_use_equal_immutable_options(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No native library is needed to verify backend selection and option merging."""
    captured = []
    sentinel = SimpleNamespace()

    def construct(spec, definition, geometry, *, reid):
        captured.append((spec, definition, geometry, reid))
        return sentinel

    monkeypatch.setattr(factory, "_create_native_tracker", construct)
    monkeypatch.setattr(factory, "_bind_and_validate_capabilities", lambda value, _: value)
    expected = TrackerSpec(name, backend="cpp", options=(("max_age", 30),))
    assert create_tracker(name, backend="cpp", max_age=30) is sentinel
    assert create_tracker(expected) is sentinel
    assert captured[0] == captured[1]


def test_collection_options_are_frozen_and_unknown_algorithm_fields_are_rejected() -> None:
    """Specs freeze authored containers; construction validates the algorithm schema."""
    nested = {"schedule": [1, 2, {"enabled": True}]}
    spec = factory._resolve_spec("bytetrack", {"backend": "cpp", "options": {"fixture": nested}})
    nested["schedule"].append(3)
    assert spec.option_dict["fixture"] == (("schedule", (1, 2, (("enabled", True),))),)
    assert hash(spec)
    with pytest.raises(TypeError, match="fixture"):
        create_tracker(spec)


@pytest.mark.parametrize("name", ("bytetrack", "occluboost"))
@pytest.mark.parametrize("option", sorted(factory._REID_MODEL_OPTIONS))
def test_factory_model_options_remain_outside_tracker_ownership(name: str, option: str) -> None:
    """A shorthand call must not hide a detector or appearance model in options."""
    message = "tracker-algorithm options only" if name == "occluboost" else "does not accept ReID model options"

    with pytest.raises(ValueError, match=message):
        create_tracker(name, options={option: Path("models/custom.pt")})


@pytest.mark.parametrize("name", ["botsort", "occluboost"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("configured", [False, True])
def test_factory_passes_live_reid_separately_and_preserves_lazy_inference(
    name: str, backend: str, configured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    import boxmot.reid.factory as reid_factory
    from tests.unit.native.trackers.test_native_live_embeddings import _FakeLibrary
    from tests.unit.trackers.box.test_live_embeddings import _detections, _frame

    encoder = RecordingEncoder(torch.eye(2, 3).numpy())
    config = ReIDConfig(model=Path("unused-model.pt"), allow_download=False)
    calls = []

    def construct(received: ReIDConfig):
        calls.append(received)
        return encoder

    monkeypatch.setattr(reid_factory, "create_reid_encoder", construct)
    if backend == "cpp":
        module = importlib.import_module(f"boxmot.trackers.{name}.native")
        monkeypatch.setattr(module, f"get_{name}_library", lambda: _FakeLibrary())
    original = TrackerSpec(name, backend=backend, options=(("use_cmc", False),))
    tracker = create_tracker(original, reid=config if configured else encoder)
    try:
        assert calls == []
        tracker.update(_detections(embeddings=torch.eye(2, 3)), _frame())
        assert calls == []
        tracker.update(_detections(), _frame())
        assert calls == ([config] if configured else [])
        assert len(encoder.calls) == 1
        assert original.option_dict == {"use_cmc": False}
    finally:
        if backend == "cpp":
            tracker.close()


@pytest.mark.parametrize("backend", ("python", "cpp"))
def test_unknown_algorithm_keywords_are_rejected_before_any_native_library_load(backend: str) -> None:
    with pytest.raises(TypeError, match="max_gae"):
        create_tracker("bytetrack", backend=backend, max_gae=30)


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"backend": "cuda"}, ValueError, "Unknown tracker backend"),
        ({"geometry": "polygon"}, ValueError, "Unknown tracker geometry"),
        ({"per_class": 1}, TypeError, "per_class must be bool"),
        ({"class_ids": [True, 1]}, TypeError, "non-negative integers"),
        ({"class_ids": [-1]}, TypeError, "non-negative integers"),
        ({"class_names": {"1": "person"}}, TypeError, "class_names must map"),
        ({"class_names": {1: ""}}, TypeError, "class_names must map"),
        ({"options": []}, TypeError, "options must be a mapping"),
        ({"options": {"geometry": "obb"}}, ValueError, "outside options"),
        ({"is_obb": True}, ValueError, "geometry="),
        ({"max_age": float("nan")}, ValueError, "non-finite"),
    ],
)
def test_shorthand_reuses_spec_and_algorithm_validation(options, error, message) -> None:
    with pytest.raises(error, match=message):
        create_tracker("bytetrack", **options)


def test_unsupported_native_tracker_and_geometry_still_fail_clearly() -> None:
    with pytest.raises(ValueError, match="Native backend is unavailable"):
        create_tracker("eagermot", backend="cpp")
    with pytest.raises(ValueError, match="does not support geometry kind"):
        create_tracker("eagermot", geometry="obb")
    with pytest.raises(ValueError, match="Native trackers do not support per_class"):
        create_tracker("bytetrack", backend="cpp", per_class=True)


@pytest.mark.parametrize("name", ("", "OccluBoost", "occluboost:cpp", "missing-tracker"))
def test_tracker_names_keep_canonical_spelling_and_registry_validation(name: str) -> None:
    with pytest.raises(ValueError):
        create_tracker(name)
