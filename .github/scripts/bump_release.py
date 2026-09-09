"""Prepare synchronized release versions without installing dependencies or committing."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
from pathlib import Path

import tomllib

VERSION_FILES = frozenset({"pyproject.toml", "boxmot/__init__.py", "uv.lock"})
BUMP_TYPES = ("major", "minor", "patch")
_STABLE_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def _git(root: Path, *arguments: str) -> str:
    """Run a read-only Git command in the candidate checkout."""
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True).stdout


def _version_assignment(source: str) -> ast.Constant:
    """Locate exactly one literal assignment to the runtime version."""
    module = ast.parse(source)
    bindings = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and node.id == "__version__" and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    assignments = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if len(bindings) != 1 or len(assignments) != 1 or assignments[0] not in module.body:
        raise ValueError("boxmot/__init__.py must contain exactly one top-level __version__ assignment.")
    assignment = assignments[0]
    if isinstance(assignment, ast.Assign) and len(assignment.targets) != 1:
        raise ValueError("The __version__ assignment must not assign other names.")
    if not isinstance(assignment.value, ast.Constant) or not isinstance(assignment.value.value, str):
        raise ValueError("The __version__ assignment must contain a literal version string.")
    return assignment.value


def _versions(root: Path) -> tuple[str, str, str]:
    """Read the project, runtime, and local editable lock-record versions."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project.get("name") != "boxmot":
        raise ValueError("The release project must be named boxmot.")
    runtime = _version_assignment((root / "boxmot/__init__.py").read_text(encoding="utf-8")).value
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    packages = [package for package in lock.get("package", []) if package.get("name") == "boxmot"]
    if len(packages) != 1 or packages[0].get("source") != {"editable": "."}:
        raise ValueError("uv.lock must contain exactly one local editable boxmot package.")
    versions = (project.get("version"), runtime, packages[0].get("version"))
    for name, version in zip(("project", "runtime", "lock"), versions, strict=True):
        if not isinstance(version, str) or _STABLE_VERSION.fullmatch(version) is None:
            raise ValueError(f"The {name} version must use stable major.minor.patch format, got {version!r}.")
    if len(set(versions)) != 1:
        raise ValueError(f"Project, runtime, and lock versions must agree, got {versions!r}.")
    return versions


def bump_release(root: Path, bump_type: str) -> tuple[str, str]:
    """Bump a clean candidate checkout and validate its three version-bearing files."""
    root = Path(root).resolve()
    if bump_type not in BUMP_TYPES:
        raise ValueError(f"Unknown release bump {bump_type!r}; expected one of {BUMP_TYPES!r}.")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Release preparation requires a clean checkout.")
    old_version = _versions(root)[0]
    components = [int(component) for component in old_version.split(".")]
    index = BUMP_TYPES.index(bump_type)
    components[index] += 1
    components[index + 1 :] = [0] * (2 - index)
    new_version = ".".join(map(str, components))

    subprocess.run(["uv", "version", "--bump", bump_type, "--no-sync"], cwd=root, check=True)

    path = root / "boxmot/__init__.py"
    source = path.read_bytes()
    value = _version_assignment(source.decode("utf-8"))
    lines = source.splitlines(keepends=True)
    # AST columns count UTF-8 bytes, including any non-ASCII text before a value.
    start = sum(map(len, lines[: value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[: value.end_lineno - 1])) + value.end_col_offset
    path.write_bytes(source[:start] + f'"{new_version}"'.encode("utf-8") + source[end:])

    actual_version = _versions(root)[0]
    if actual_version != new_version:
        raise ValueError(f"Expected a {bump_type} bump to {new_version}, got {actual_version}.")
    changed = set(_git(root, "diff", "--name-only", "-z", "HEAD").split("\0"))
    changed.update(_git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0"))
    changed.discard("")
    if changed != VERSION_FILES:
        raise ValueError(f"Release preparation must change only {sorted(VERSION_FILES)!r}, got {sorted(changed)!r}.")
    return old_version, new_version


def main() -> None:
    """Prepare the current checkout and expose candidate versions to GitHub Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump", choices=BUMP_TYPES, required=True)
    args = parser.parse_args()
    old_version, version = bump_release(Path.cwd(), args.bump)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(f"old_version={old_version}\nversion={version}\n")
    print(f"Prepared BoxMOT {old_version} -> {version}")


if __name__ == "__main__":
    main()
