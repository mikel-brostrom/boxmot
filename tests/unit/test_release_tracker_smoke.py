"""Verify that release smoke catches unusable public tracker installations."""

from __future__ import annotations

from pathlib import Path

import pytest

import boxmot
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common import config as tracker_config
from boxmot.trackers.common.appearance.live import LiveReIDMixin
from tests.ci import release_contract


@pytest.fixture
def forbid_reid_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail before an omitted synthetic embedding can construct a model."""

    def unexpected_model_loading(self: LiveReIDMixin) -> None:
        raise AssertionError("Release smoke must supply embeddings without constructing a ReID model")

    monkeypatch.setattr(LiveReIDMixin, "_get_live_reid_model", unexpected_model_loading)
    monkeypatch.setattr(LiveReIDMixin, "_get_live_reid_encoder", unexpected_model_loading)


@pytest.mark.parametrize("failure", ("missing-export", "missing-module", "missing-class"))
def test_tracker_smoke_resolves_lazy_exports_despite_unchanged_public_names(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    public_name = "BoostTrack"
    module_name, _ = boxmot._EXPORTS[public_name]
    monkeypatch.delitem(boxmot.__dict__, public_name, raising=False)
    if failure == "missing-export":
        monkeypatch.delitem(boxmot._EXPORTS, public_name)
    elif failure == "missing-module":
        monkeypatch.setitem(boxmot._EXPORTS, public_name, ("boxmot.missing_release_tracker", public_name))
    else:
        monkeypatch.setitem(boxmot._EXPORTS, public_name, (module_name, "MissingReleaseTracker"))

    assert boxmot.__all__ == release_contract.EXPECTED_PUBLIC_API
    expected_error = ModuleNotFoundError if failure == "missing-module" else AttributeError
    with pytest.raises(expected_error):
        release_contract.check_tracker_imports()


def test_tracker_smoke_rejects_a_public_name_resolving_to_another_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(boxmot, "BoostTrack", boxmot.ByteTrack)

    with pytest.raises(AssertionError, match=r"boxmot.BoostTrack resolves to"):
        release_contract.check_tracker_imports()


def test_tracker_smoke_propagates_factory_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_factory(spec: TrackerSpec) -> None:
        raise RuntimeError(f"Cannot construct packaged tracker {spec.name}")

    monkeypatch.setattr(boxmot, "create_tracker", broken_factory)

    with pytest.raises(RuntimeError, match="Cannot construct packaged tracker boosttrack"):
        release_contract.check_tracker_api()


def test_tracker_smoke_requires_packaged_tracker_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tracker_config, "TRACKER_CONFIGS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=r"Tracker config not found: .*boosttrack.yaml"):
        release_contract.check_tracker_api()


@pytest.mark.usefixtures("forbid_reid_loading")
@pytest.mark.parametrize(
    ("name", "public_name", "geometry"),
    [
        (name, public_name, geometry)
        for name, public_name in release_contract.EXPECTED_TRACKERS
        for geometry in (("aabb",) if name in {"eagermot", "maf_hda"} else ("aabb", "obb"))
    ],
)
def test_tracker_smoke_runs_real_default_trackers_without_models(name: str, public_name: str, geometry: str) -> None:
    release_contract.check_tracker_tracking(name, getattr(boxmot, public_name), geometry)


def test_tracker_api_smoke_covers_every_geometry_and_runs_the_packed_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, type, str]] = []

    def record_tracking(name: str, tracker_class: type, geometry: str) -> None:
        calls.append((name, tracker_class, geometry))

    monkeypatch.setattr(release_contract, "check_tracker_tracking", record_tracking)

    # Keep the direct ByteTrack NumPy smoke real while checking its dispatcher.
    release_contract.check_tracker_api()

    assert calls == [
        (name, getattr(boxmot, public_name), geometry)
        for name, public_name in release_contract.EXPECTED_TRACKERS
        for geometry in (("aabb",) if name in {"eagermot", "maf_hda"} else ("aabb", "obb"))
    ]
