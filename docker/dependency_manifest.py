"""Copy Docker dependency metadata with only BoxMOT's release version normalized.

The output is for dependency-only syncs. Package builds must use the original
metadata so the installed BoxMOT distribution retains its actual version.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import tomllib

CACHE_VERSION = "0.0.0"
_TABLE_HEADER = re.compile(rb"(?m)^[ \t]*\[\[?[^\r\n]+?\]\]?[ \t]*(?:#[^\r\n]*)?\r?$")
_VERSION_LINE = re.compile(
    rb"(?m)^[ \t]*version[ \t]*=[ \t]*(?P<quote>[\"'])(?P<value>[^\"'\r\n]*)(?P=quote)"
    rb"[ \t]*(?:#[^\r\n]*)?\r?$"
)


def _normalize_version(source: bytes, header: bytes, index: int, expected: dict) -> bytes:
    """Replace one scalar value in a canonical TOML table without reserializing."""
    tables = list(_TABLE_HEADER.finditer(source))
    selected = [
        position
        for position, table in enumerate(tables)
        if table.group().split(b"#", 1)[0].strip() == header
    ]
    if index >= len(selected):
        raise ValueError(f"Missing canonical {header.decode()} table in dependency metadata.")
    position = selected[index]
    start = tables[position].end()
    end = tables[position + 1].start() if position + 1 < len(tables) else len(source)
    versions = list(_VERSION_LINE.finditer(source, start, end))
    if len(versions) != 1:
        raise ValueError(f"Expected one scalar version assignment in {header.decode()}.")
    value = versions[0]
    normalized = source[: value.start("value")] + CACHE_VERSION.encode() + source[value.end("value") :]
    # TOML strings can contain text resembling headers or assignments. Fail
    # closed if matching that text would change anything beyond the root version.
    if tomllib.loads(normalized.decode("utf-8")) != expected:
        raise ValueError("Normalizing dependency metadata would change fields other than the BoxMOT version.")
    return normalized


def normalize_dependency_manifest(pyproject: Path, lockfile: Path, output: Path) -> None:
    """Write version-independent copies after validating the editable root pair."""
    project_source = pyproject.read_bytes()
    lock_source = lockfile.read_bytes()
    project_document = tomllib.loads(project_source.decode("utf-8"))
    lock_document = tomllib.loads(lock_source.decode("utf-8"))
    project = project_document.get("project")
    if not isinstance(project, dict) or project.get("name") != "boxmot":
        raise ValueError("pyproject.toml must declare the boxmot project.")
    version = project.get("version")
    dynamic = project.get("dynamic", [])
    if not isinstance(dynamic, list):
        raise ValueError("pyproject.toml dynamic metadata must be a list of field names.")
    if not isinstance(version, str) or not version or "version" in dynamic:
        raise ValueError("pyproject.toml must declare a nonempty static project version.")
    packages = lock_document.get("package")
    if not isinstance(packages, list) or not all(isinstance(package, dict) for package in packages):
        raise ValueError("uv.lock must declare package records.")
    roots = [index for index, package in enumerate(packages) if package.get("name") == "boxmot"]
    editable_roots = [index for index, package in enumerate(packages) if package.get("source") == {"editable": "."}]
    if len(roots) != 1 or editable_roots != roots:
        raise ValueError("uv.lock must contain exactly one local editable boxmot root package.")
    root = packages[roots[0]]
    if root.get("version") != version:
        raise ValueError("The project and local editable boxmot lock versions must match.")

    project["version"] = root["version"] = CACHE_VERSION
    normalized_project = _normalize_version(project_source, b"[project]", 0, project_document)
    normalized_lock = _normalize_version(lock_source, b"[[package]]", roots[0], lock_document)
    targets = (output / "pyproject.toml", output / "uv.lock")
    for target in targets:
        for original in (pyproject, lockfile):
            if target.resolve() == original.resolve() or (target.exists() and target.samefile(original)):
                raise ValueError("Dependency metadata output must not overwrite either original input.")
    output.mkdir(parents=True, exist_ok=True)
    for target, content in zip(targets, (normalized_project, normalized_lock), strict=True):
        target.write_bytes(content)


def main() -> None:
    """Prepare dependency-only metadata from the current checkout or explicit paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        normalize_dependency_manifest(args.pyproject, args.lockfile, args.output)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
