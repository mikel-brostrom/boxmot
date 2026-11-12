"""Exercise optional backend metadata builds in a fresh CI environment."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="Dependency configuration checks use Python 3.11 or newer.")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _backend_archive(directory: Path) -> tuple[Path, Path]:
    """Create an optional source package whose metadata needs a local build dependency."""
    wheels = directory / "wheels"
    wheels.mkdir()
    backend = (
        "from pathlib import Path\n"
        "def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):\n"
        '    path = Path(metadata_directory) / "edgetam-1.0.dist-info"\n'
        "    path.mkdir()\n"
        '    (path / "METADATA").write_text("Metadata-Version: 2.3\\nName: edgetam\\nVersion: 1.0\\n")\n'
        "    return path.name\n"
    )
    with zipfile.ZipFile(wheels / "fixture_backend-1.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("fixture_backend.py", backend)
        wheel.writestr(
            "fixture_backend-1.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: fixture-backend\nVersion: 1.0\n",
        )
        wheel.writestr(
            "fixture_backend-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr("fixture_backend-1.0.dist-info/RECORD", "")

    archive = directory / "edgetam-1.0.tar.gz"
    content = b'[build-system]\nrequires = ["fixture-backend==1.0"]\nbuild-backend = "fixture_backend"\n'
    with tarfile.open(archive, "w:gz") as source:
        member = tarfile.TarInfo("edgetam-1.0/pyproject.toml")
        member.size = len(content)
        source.addfile(member, io.BytesIO(content))
    return archive, wheels


def test_locked_sync_bootstraps_unselected_backend_metadata(tmp_path: Path) -> None:
    """An unselected EdgeTAM group must not need build tools in the project environment."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("Dependency metadata validation requires the repository's pinned uv executable.")
    archive, wheels = _backend_archive(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "metadata-probe"\nversion = "1.0"\nrequires-python = ">=3.10"\n'
        f'[dependency-groups]\nedgetam = ["edgetam @ {archive.as_uri()}"]\n',
        encoding="utf-8",
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    environment.update(
        UV_CACHE_DIR=str(tmp_path / "lock-cache"),
        UV_PROJECT_ENVIRONMENT=str(tmp_path / "unused-venv"),
        UV_PYTHON=sys.executable,
        UV_PYTHON_DOWNLOADS="never",
        UV_OFFLINE="1",
    )

    def run_uv(*arguments: str) -> subprocess.CompletedProcess[str]:
        """Resolve solely against the fixture's local build wheel."""
        return subprocess.run(
            [uv, *arguments, "--no-index", "--find-links", str(wheels)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    locked = run_uv("lock")
    assert locked.returncode == 0, locked.stderr
    before = (tmp_path / "uv.lock").read_bytes()

    # Validate an existing lock with the repository's isolation policy and no
    # cached metadata. The optional group remains unselected, as in CPU CI.
    settings = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]
    without_isolation = settings.get("no-build-isolation-package", [])
    with pyproject.open("a", encoding="utf-8") as project:
        project.write(f"[tool.uv]\nno-build-isolation-package = {without_isolation!r}\n")
    environment["UV_CACHE_DIR"] = str(tmp_path / "fresh-cache")

    result = run_uv("sync", "--locked", "--dry-run", "--no-default-groups")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "uv.lock").read_bytes() == before
    assert not (tmp_path / "unused-venv").exists()
    assert not (tmp_path / ".venv").exists()
