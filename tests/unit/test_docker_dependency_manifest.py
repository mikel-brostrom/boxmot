"""Validate release-stable dependency metadata without weakening uv's lock checks."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="Docker metadata preparation uses Python 3.11 or newer.")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "dependency_manifest.py"
SPEC = importlib.util.spec_from_file_location("dependency_manifest", SCRIPT)
dependency_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dependency_manifest)


@pytest.fixture
def metadata(tmp_path: Path) -> tuple[Path, Path]:
    """Include same-version third parties, nested metadata, comments, and CRLFs."""
    pyproject = tmp_path / "pyproject.toml"
    lockfile = tmp_path / "uv.lock"
    pyproject.write_bytes(
        b'# Keep metadata formatting.\r\n[project]\r\nname = "boxmot"\r\n'
        b"version \t = '24.0.0'  # Actual release\r\n"
        b'dependencies = ["example==24.0.0"]\r\n'
        b'[tool.example]\r\nversion = "24.0.0"\r\n'
    )
    lockfile.write_bytes(
        b'version = 1\nrevision = 3\nrequires-python = ">=3.11"\n'
        b'[[package]]\nname = "example"\nversion = "24.0.0"\n'
        b'source = { registry = "https://pypi.org/simple" }\n'
        b'sdist = { url = "https://example.invalid/example.tar.gz", hash = "sha256:abc123" }\n'
        b'[[package]]\nname = "boxmot"\nversion = "24.0.0" # Root only\n'
        b'source = { editable = "." }\n'
        b'dependencies = [{ name = "example", marker = "sys_platform == \'linux\'" }]\n'
        b'[package.metadata]\nrequires-dist = [{ name = "example", specifier = "==24.0.0" }]\n'
    )
    return pyproject, lockfile


def _normalize(metadata: tuple[Path, Path], output: Path) -> tuple[bytes, bytes]:
    """Return exactly the files consumed by Docker's dependency stages."""
    dependency_manifest.normalize_dependency_manifest(*metadata, output)
    return (output / "pyproject.toml").read_bytes(), (output / "uv.lock").read_bytes()


def test_only_the_two_root_versions_change(metadata: tuple[Path, Path], tmp_path: Path) -> None:
    before = tuple(path.read_bytes() for path in metadata)

    result = _normalize(metadata, tmp_path / "dependencies")

    assert result == (
        before[0].replace(b"version \t = '24.0.0'", b"version \t = '0.0.0'", 1),
        before[1].replace(b'version = "24.0.0" # Root only', b'version = "0.0.0" # Root only', 1),
    )
    assert tuple(path.read_bytes() for path in metadata) == before


@pytest.mark.parametrize("release", ("24.0.1", "24.1.0", "25.0.0"))
def test_release_only_bumps_have_identical_outputs(
    metadata: tuple[Path, Path], tmp_path: Path, release: str
) -> None:
    baseline = _normalize(metadata, tmp_path / "before")
    pyproject, lockfile = metadata
    pyproject.write_bytes(
        pyproject.read_bytes().replace(b"version \t = '24.0.0'", f"version \t = '{release}'".encode(), 1)
    )
    lockfile.write_bytes(
        lockfile.read_bytes().replace(
            b'version = "24.0.0" # Root only', f'version = "{release}" # Root only'.encode(), 1
        )
    )

    assert _normalize(metadata, tmp_path / "after") == baseline


@pytest.mark.parametrize(
    "index,old,new",
    (
        (0, b"example==24.0.0", b"example>=24.0.0"),
        (0, b"[tool.example]", b"[tool.another]"),
        (1, b'name = "example"\nversion = "24.0.0"', b'name = "example"\nversion = "24.0.1"'),
        (1, b"https://pypi.org/simple", b"https://registry.example.invalid/simple"),
        (1, b"sha256:abc123", b"sha256:def456"),
        (1, b"sys_platform == 'linux'", b"sys_platform == 'darwin'"),
        (1, b'specifier = "==24.0.0"', b'specifier = ">=24.0.0"'),
    ),
)
def test_dependency_metadata_changes_invalidate_outputs(
    metadata: tuple[Path, Path], tmp_path: Path, index: int, old: bytes, new: bytes
) -> None:
    baseline = _normalize(metadata, tmp_path / "before")
    original = metadata[index].read_bytes()
    assert old in original
    metadata[index].write_bytes(original.replace(old, new, 1))

    assert _normalize(metadata, tmp_path / "after") != baseline


@pytest.mark.parametrize(
    "index,old,new,error",
    (
        (0, b'name = "boxmot"', b'name = "other"', "boxmot project"),
        (0, b"version \t = '24.0.0'", b"version \t = 24", "static project version"),
        (0, b"version \t = '24.0.0'", b"version \t = ''", "static project version"),
        (0, b"[tool.example]", b'dynamic = ["version"]\r\n[tool.example]', "static project version"),
        (1, b'name = "boxmot"', b'name = "other"', "exactly one local editable boxmot"),
        (1, b'editable = "."', b'editable = "../boxmot"', "exactly one local editable boxmot"),
        (1, b'editable = "."', b'virtual = "."', "exactly one local editable boxmot"),
        (1, b'version = "24.0.0" # Root only', b'version = "24.0.1" # Root only', "versions must match"),
        (1, b"[[package]]", b"[package]", "Cannot overwrite a value"),
    ),
)
def test_invalid_inputs_fail_before_writing(
    metadata: tuple[Path, Path], tmp_path: Path, index: int, old: bytes, new: bytes, error: str
) -> None:
    metadata[index].write_bytes(metadata[index].read_bytes().replace(old, new, 1))
    before = tuple(path.read_bytes() for path in metadata)
    output = tmp_path / "dependencies"

    with pytest.raises(ValueError, match=error):
        _normalize(metadata, output)

    assert not output.exists()
    assert tuple(path.read_bytes() for path in metadata) == before


@pytest.mark.parametrize("name", ("boxmot", "other"))
def test_duplicate_root_records_are_rejected(metadata: tuple[Path, Path], tmp_path: Path, name: str) -> None:
    lockfile = metadata[1]
    lockfile.write_bytes(
        lockfile.read_bytes()
        + f'\n[[package]]\nname = "{name}"\nversion = "24.0.0"\nsource = {{ editable = "." }}\n'.encode()
    )

    with pytest.raises(ValueError, match="exactly one local editable boxmot"):
        _normalize(metadata, tmp_path / "dependencies")


@pytest.mark.parametrize("alias", ("same-directory", "symlink", "hardlink"))
def test_outputs_cannot_overwrite_inputs(metadata: tuple[Path, Path], tmp_path: Path, alias: str) -> None:
    before = tuple(path.read_bytes() for path in metadata)
    output = tmp_path if alias == "same-directory" else tmp_path / "dependencies"
    if alias != "same-directory":
        output.mkdir()
        target = output / "pyproject.toml"
        if alias == "symlink":
            target.symlink_to(metadata[0])
        else:
            target.hardlink_to(metadata[0])

    with pytest.raises(ValueError, match="must not overwrite"):
        _normalize(metadata, output)

    assert tuple(path.read_bytes() for path in metadata) == before


def test_header_text_in_multiline_strings_cannot_change_unrelated_fields(
    metadata: tuple[Path, Path], tmp_path: Path
) -> None:
    pyproject = metadata[0]
    pyproject.write_bytes(b'example = """\n[project]\nversion = \'24.0.0\'\n"""\n' + pyproject.read_bytes())

    with pytest.raises(ValueError, match="fields other than the BoxMOT version"):
        _normalize(metadata, tmp_path / "dependencies")


def test_cli_accepts_metadata_from_its_working_directory(metadata: tuple[Path, Path], tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "cli-output")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli-output" / "pyproject.toml").read_bytes() == _normalize(metadata, tmp_path / "direct")[0]


@pytest.mark.parametrize(
    "profile",
    (
        ("--extra", "cpu", "--extra", "yolo", "--extra", "trackeval"),
        ("--extra", "cu130", "--extra", "yolo", "--extra", "trackeval"),
        ("--only-group", "service-runtime"),
        ("--extra", "cu130", "--extra", "service", "--extra", "service-gpu"),
    ),
)
def test_repository_dependency_exports_are_unchanged_offline(tmp_path: Path, profile: tuple[str, ...]) -> None:
    """Compare full locked dependency graphs, including hashes, for every image."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("Docker lock validation requires the repository's pinned uv executable.")
    output = tmp_path / "dependencies"
    _normalize((REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"), output)
    original = tmp_path / "original"
    original.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (original / name).write_bytes((REPO_ROOT / name).read_bytes())
    environment = {
        **os.environ,
        "UV_OFFLINE": "1",
        "UV_PYTHON": sys.executable,
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "unused-venv"),
    }
    exports = []
    for directory in (original, output):
        before = tuple((directory / name).read_bytes() for name in ("pyproject.toml", "uv.lock"))
        result = subprocess.run(
            [uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--no-header", *profile],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert tuple((directory / name).read_bytes() for name in ("pyproject.toml", "uv.lock")) == before
        exports.append(result.stdout)

    assert exports[0] == exports[1]
    assert "--hash=sha256:" in exports[0]
    assert not (tmp_path / "unused-venv").exists()
    assert not (output / ".venv").exists()


def test_normalized_manifest_passes_uv_locked_sync_offline(tmp_path: Path) -> None:
    """Check freshness with a real third-party dependency and no registry cache."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("Docker lock validation requires the repository's pinned uv executable.")
    original = tmp_path / "original"
    original.mkdir()
    example = tmp_path / "example"
    example.mkdir()
    (example / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "24.0.0"\nrequires-python = ">=3.11"\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n',
        encoding="utf-8",
    )
    (original / "pyproject.toml").write_text(
        '[project]\nname = "boxmot"\nversion = "24.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["example==24.0.0"]\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.uv]\npreview = true\nrequired-version = "==0.12.4"\n'
        '[tool.uv.sources]\nexample = { path = "../example" }\n',
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "UV_OFFLINE": "1",
        "UV_PYTHON": sys.executable,
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "unused-venv"),
    }
    subprocess.run([uv, "lock"], cwd=original, env=environment, capture_output=True, text=True, check=True)
    output = tmp_path / "dependencies"
    normalized = _normalize((original / "pyproject.toml", original / "uv.lock"), output)
    result = subprocess.run(
        [uv, "sync", "--locked", "--dry-run", "--no-dev", "--no-install-package", "boxmot"],
        cwd=output,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert tuple((output / name).read_bytes() for name in ("pyproject.toml", "uv.lock")) == normalized
    assert not (tmp_path / "unused-venv").exists()
    assert not (output / ".venv").exists()
