from pathlib import Path

from boxmot.resources.paths import resolve_model_path


def test_resolve_model_path_falls_back_for_bare_name(tmp_path: Path) -> None:
    weights = tmp_path / "models"
    weights.mkdir()

    assert resolve_model_path("encoder.pt", default_dir=weights) == weights / "encoder.pt"


def test_resolve_model_path_preserves_explicit_relative_path(tmp_path: Path) -> None:
    explicit = tmp_path / "nested" / "encoder.pt"

    assert resolve_model_path(explicit, default_dir=tmp_path / "models") == explicit


def test_resolve_model_path_matches_existing_name_case_insensitively(tmp_path: Path) -> None:
    weights = tmp_path / "models"
    weights.mkdir()
    existing = weights / "Encoder.PT"
    existing.touch()

    resolved = resolve_model_path("encoder.pt", default_dir=weights)

    assert resolved.exists()
    assert resolved.samefile(existing)


def test_resolve_model_path_preserves_selector_under_existing_file(tmp_path: Path) -> None:
    profile = tmp_path / "detector.yaml"
    profile.write_text("id: detector\n", encoding="utf-8")
    selector = profile / "default"

    assert resolve_model_path(selector, default_dir=tmp_path / "models") == selector
