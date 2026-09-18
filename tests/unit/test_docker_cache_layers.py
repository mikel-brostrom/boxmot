"""Protect the Docker boundaries between system, dependency, and application layers."""

from __future__ import annotations

import shlex
from collections.abc import Iterator

import pytest

from tests.unit.test_docker_native_runtime import _stages

Stages = dict[str, tuple[str, list[str]]]
PROFILES = {
    "cli-cpu": ({"cpu", "yolo", "trackeval"}, set()),
    "cli-gpu": ({"cu130", "yolo", "trackeval"}, set()),
    "service-cpu": (set(), {"service-runtime"}),
    "service-gpu": ({"cu130", "service", "service-gpu"}, set()),
}


def _lineage(stages: Stages, name: str) -> Iterator[str]:
    """Follow FROM ancestry; copying files does not inherit a stage's layers."""
    while name in stages:
        yield name
        name = stages[name][0]


def _instructions(stages: Stages, name: str) -> list[str]:
    """Return inherited instructions in build order."""
    return [instruction for stage in reversed(list(_lineage(stages, name))) for instruction in stages[stage][1]]


def _copies(instructions: list[str]) -> Iterator[tuple[str | None, tuple[str, ...], str]]:
    """Read each COPY's origin, source paths, and destination, ignoring other flags."""
    for instruction in instructions:
        if not instruction.startswith("COPY "):
            continue
        words = shlex.split(instruction)[1:]
        origin = next((word.removeprefix("--from=") for word in words if word.startswith("--from=")), None)
        paths = [word for word in words if not word.startswith("--")]
        yield origin, tuple(paths[:-1]), paths[-1]


def _command(instructions: list[str], *command: str) -> tuple[int, list[str]]:
    """Locate a unique shell command without mistaking comments for instructions."""
    matches = []
    for index, instruction in enumerate(instructions):
        if not instruction.startswith("RUN "):
            continue
        words = shlex.split(instruction)
        if any(words[offset : offset + len(command)] == list(command) for offset in range(len(words))):
            matches.append((index, words))
    assert len(matches) == 1, f"Expected one {' '.join(command)} command, found {len(matches)}"
    return matches[0]


def _values(words: list[str], option: str) -> set[str]:
    """Collect values for a repeated CLI option."""
    return {words[index + 1] for index, word in enumerate(words[:-1]) if word == option}


def test_system_package_layers_do_not_depend_on_project_files() -> None:
    """No metadata or application copy may precede apt, including in parent stages."""
    stages = _stages()
    checked = []
    for name, (parent, instructions) in stages.items():
        for index, instruction in enumerate(instructions):
            if instruction.startswith("RUN ") and "apt-get install" in instruction:
                copies = list(_copies(_instructions(stages, parent) + instructions[:index]))
                assert all(origin == "uv" for origin, _, _ in copies), (name, copies)
                checked.append(name)
    assert checked


def test_native_compilation_depends_only_on_native_source() -> None:
    """Release metadata and Python dependencies cannot invalidate native compilation."""
    copies = list(_copies(_instructions(_stages(), "native-cli-builder")))
    assert {source for origin, sources, _ in copies if origin is None for source in sources} == {"boxmot/native/cpp"}
    assert {origin for origin, _, _ in copies if origin is not None} == {"uv"}


@pytest.mark.parametrize("target", PROFILES)
def test_dependency_stages_consume_only_normalized_metadata(target: str) -> None:
    """Only normalized files cross into dependency builders; no raw metadata layers do."""
    stages = _stages()
    name = f"{target}-dependencies"
    assert "dependency-manifests" not in _lineage(stages, name)
    instructions = _instructions(stages, name)
    copies = [copy for copy in _copies(instructions) if copy[0] != "uv"]
    assert copies == [("dependency-manifests", ("/opt/boxmot-dependencies/",), "./")]
    _, words = _command(instructions, "uv", "sync")
    assert {"--locked", "--no-dev"} <= set(words)
    assert _values(words, "--no-install-package") == {"boxmot"}
    extras, groups = PROFILES[target]
    assert _values(words, "--extra") == extras
    assert _values(words, "--only-group") == groups


@pytest.mark.parametrize("target", PROFILES)
def test_runtime_dependency_copy_precedes_application_layer(target: str) -> None:
    """Runtime images retain the dependency-only venv as a separate reusable layer."""
    instructions = _instructions(_stages(), target)
    copies = list(_copies(instructions))
    venv = [copy for copy in copies if copy[2] == "/opt/boxmot/.venv"]
    assert venv == [(f"{target}-dependencies", ("/opt/boxmot/.venv",), "/opt/boxmot/.venv")]
    dependency_index = next(index for index, row in enumerate(instructions) if f"--from={target}-dependencies " in row)
    if target.startswith("cli-"):
        application_index, words = _command(instructions, "uv", "pip", "install")
        assert {"--no-deps", "--no-cache", "/wheels/*.whl"} <= set(words)
        assert _values(words, "--python") == {"/opt/boxmot/.venv/bin/python"}
        mounts = [
            dict(item.split("=", 1) for item in word.removeprefix("--mount=").split(","))
            for word in words
            if word.startswith("--mount=")
        ]
        assert {"type": "bind", "from": "cli-wheel-builder", "source": "/wheels", "target": "/wheels"} in mounts
        assert not any(origin is None for origin, _, _ in copies)
        assert not any("UV_COMPILE_BYTECODE=1" in row or "--compile-bytecode" in row for row in instructions)
    else:
        assert [copy for copy in copies if copy[0] is None] == [(None, ("boxmot",), "/opt/boxmot/boxmot")]
        application_index = next(
            index
            for index, row in enumerate(instructions)
            if row.startswith("COPY ") and " boxmot /opt/boxmot/boxmot" in row
        )
    assert dependency_index < application_index
    assert not any(row.startswith("RUN ") and "uv sync" in row for row in instructions)


def test_shared_wheel_builder_uses_real_metadata_without_dependency_environments() -> None:
    """The application wheel retains the release version and combines native artifacts once."""
    instructions = _instructions(_stages(), "cli-wheel-builder")
    copies = list(_copies(instructions))
    context_sources = {source for origin, sources, _ in copies if origin is None for source in sources}
    assert {"pyproject.toml", "README.md", "LICENSE", "boxmot"} <= context_sources
    assert {origin for origin, _, _ in copies if origin is not None} == {"uv", "native-cli-builder"}
    assert not any(".venv" in source for _, sources, _ in copies for source in sources)
    _, words = _command(instructions, "uv", "build")
    assert {"--wheel", "--no-sources"} <= set(words)
