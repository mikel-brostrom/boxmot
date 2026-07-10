from __future__ import annotations

import difflib
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__fspath__"):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return cleaned.strip("_") or "item"


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found in subprocess stdout")

def _count_changed_lines(before: str, after: str) -> tuple[int, int]:
    diff = difflib.ndiff(before.splitlines(), after.splitlines())
    added = sum(1 for line in diff if line.startswith("+ "))
    removed = sum(1 for line in diff if line.startswith("- "))
    return added, removed

def _workspace_copy_ignore(src: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    common = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "gepa",
        "models",
        "ray",
        "runs",
    }
    for name in names:
        if name in common:
            ignored.add(name)

    return ignored

def _terminate_subprocess_tree(
    proc: subprocess.Popen[str],
    *,
    graceful: bool,
    wait_timeout: float = 5.0,
) -> tuple[str, str | None]:
    """Terminate a subprocess and any children in its process group, then reap it."""
    if os.name == "nt":
        terminator = getattr(proc, "terminate" if graceful else "kill", None)
        if callable(terminator):
            try:
                terminator()
            except ProcessLookupError:
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM if graceful else signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        return proc.communicate(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        if not graceful:
            raise

        if os.name == "nt":
            killer = getattr(proc, "kill", None)
            if callable(killer):
                try:
                    killer()
                except ProcessLookupError:
                    pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return proc.communicate(timeout=wait_timeout)
