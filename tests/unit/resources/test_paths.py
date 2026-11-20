from pathlib import Path

import boxmot.resources.paths as resource_paths
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


def test_installed_models_resolve_to_user_cache_without_filesystem_writes(tmp_path, monkeypatch) -> None:
    """A read-only site-packages tree must never be the download destination."""
    installation = tmp_path / "read-only" / "site-packages"
    module_file = installation / "boxmot" / "resources" / "paths.py"
    module_file.parent.mkdir(parents=True)
    module_file.touch()
    working_directory = tmp_path / "application"
    working_directory.mkdir()
    cache = tmp_path / "user-cache" / "boxmot"
    monkeypatch.chdir(working_directory)
    monkeypatch.setattr(resource_paths, "__file__", str(module_file))
    monkeypatch.setattr(resource_paths, "user_cache_path", lambda appname: cache)

    def refuse_directory_creation(*args, **kwargs):
        raise PermissionError("Path resolution must not create directories")

    monkeypatch.setattr(Path, "mkdir", refuse_directory_creation)

    assert resolve_model_path("encoder.pt") == cache / "models" / "encoder.pt"
    assert not cache.exists()
    assert not (installation / "models").exists()


def test_checkout_models_are_used_outside_repository_working_directory(tmp_path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "boxmot" / "resources").mkdir(parents=True)
    (checkout / "pyproject.toml").touch()
    monkeypatch.setattr(resource_paths, "__file__", str(checkout / "boxmot" / "resources" / "paths.py"))
    monkeypatch.chdir(tmp_path)

    assert resolve_model_path("encoder.pt") == checkout / "models" / "encoder.pt"


def test_working_checkout_takes_precedence_over_installed_location(tmp_path, monkeypatch) -> None:
    (tmp_path / "boxmot").mkdir()
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(resource_paths, "__file__", "/read-only/site-packages/boxmot/resources/paths.py")
    monkeypatch.chdir(tmp_path)

    assert resolve_model_path("encoder.pt") == tmp_path / "models" / "encoder.pt"


def test_explicit_relative_model_path_does_not_use_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    explicit = Path("custom") / "encoder.pt"

    assert resolve_model_path(explicit) == explicit
