from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from boxmot.utils import ROOT
from boxmot.utils import logger as LOGGER

from .constants import _PROPOSAL_VALIDATION_ATTEMPTS
from .paths import _count_changed_lines, _is_relative_to, _json_default, _slugify


def _normalize_editable_files(
    tracker: str,
    editable_files: Sequence[str | Path] | None,
) -> tuple[str, ...]:
    if editable_files:
        files = [Path(path) for path in editable_files]
    else:
        tracker_name = tracker.lower()
        tracker_file = ROOT / "boxmot" / "trackers" / "bbox" / f"{tracker_name}.py"
        tracker_dir = ROOT / "boxmot" / "trackers" / "bbox" / tracker_name
        if tracker_file.exists():
            files = [tracker_file]
        elif tracker_dir.exists():
            main_impl = tracker_dir / f"{tracker_name}.py"
            if main_impl.exists():
                files = [main_impl]
            else:
                files = sorted(path for path in tracker_dir.rglob("*.py") if path.name != "__init__.py")
        else:
            # Fallback for non-bbox trackers (e.g. hybrid)
            tracker_dir = ROOT / "boxmot" / "trackers" / tracker_name
            if not tracker_dir.exists():
                raise FileNotFoundError(f"Tracker source not found for: {tracker}")

            main_impl = tracker_dir / f"{tracker_name}.py"
            if main_impl.exists():
                files = [main_impl]
            else:
                files = sorted(path for path in tracker_dir.rglob("*.py") if path.name != "__init__.py")

    normalized: list[str] = []
    for path in files:
        abs_path = path if path.is_absolute() else (ROOT / path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Editable file not found: {abs_path}")
        if not _is_relative_to(abs_path, ROOT):
            raise ValueError(f"Editable file must live under the repository root: {abs_path}")
        normalized.append(abs_path.relative_to(ROOT).as_posix())

    if not normalized:
        raise ValueError(f"No editable files found for tracker '{tracker}'")

    return tuple(dict.fromkeys(normalized))


def _read_candidate(files: Sequence[str]) -> dict[str, str]:
    return {file_path: (ROOT / file_path).read_text(encoding="utf-8") for file_path in files}


def _validate_candidate_keys(candidate: Mapping[str, str], expected_keys: Sequence[str]) -> dict[str, str]:
    expected = tuple(expected_keys)
    actual = tuple(candidate.keys())
    if set(actual) != set(expected):
        missing = [key for key in expected if key not in candidate]
        unexpected = [key for key in actual if key not in expected]
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if unexpected:
            parts.append(f"unexpected keys: {unexpected}")
        raise ValueError("; ".join(parts))

    return {key: _raw_text(candidate[key]) for key in expected}


def _validate_candidate_content(candidate: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    for file_path, content in candidate.items():
        suffix = Path(file_path).suffix.lower()
        if suffix == ".py":
            try:
                compile(content, file_path, "exec")
            except SyntaxError as exc:
                errors.append(f"{file_path}: {exc.msg} (line {exc.lineno})")
        elif suffix in {".yaml", ".yml"}:
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as exc:
                errors.append(f"{file_path}: {exc}")
    return errors


def _candidate_change_summary(seed_candidate: Mapping[str, str], candidate: Mapping[str, str]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for file_path, current in candidate.items():
        baseline = seed_candidate[file_path]
        if current == baseline:
            continue
        added, removed = _count_changed_lines(baseline, current)
        changes.append({"path": file_path, "added": added, "removed": removed})
    return changes


def _candidate_import_modules(file_paths: Sequence[str]) -> tuple[str, ...]:
    modules: list[str] = []
    for file_path in file_paths:
        path = Path(file_path)
        if path.suffix.lower() != ".py":
            continue
        module_path = path.with_suffix("")
        parts = module_path.parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.append(".".join(parts))
    return tuple(dict.fromkeys(modules))


def _inject_validation_feedback(
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    components_to_update: Sequence[str],
    rejection_feedback: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    feedback_record = {
        "Validation Feedback": (
            "The previous proposal was rejected before benchmark evaluation. "
            "Fix the issues below and return complete replacement contents for the target file."
        ),
        "Rejected Proposal Errors": [str(message) for message in rejection_feedback],
    }
    augmented: dict[str, list[dict[str, Any]]] = {}
    for component in components_to_update:
        augmented[component] = [dict(item) for item in reflective_dataset.get(component, ())]
        augmented[component].append(feedback_record)
    return augmented


def _make_checked_candidate_proposer(
    proposal_runner: Callable[
        [dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]],
        dict[str, str],
    ],
    *,
    expected_keys: Sequence[str],
    candidate_checker: Callable[[dict[str, str]], list[str]] | None = None,
    max_attempts: int = _PROPOSAL_VALIDATION_ATTEMPTS,
) -> Callable[[dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]], dict[str, str]]:
    def proposer(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        rejection_feedback: list[str] = []
        last_proposed_updates: dict[str, str] = {}
        for attempt in range(1, max_attempts + 1):
            dataset_for_attempt = (
                _inject_validation_feedback(reflective_dataset, components_to_update, rejection_feedback)
                if rejection_feedback
                else reflective_dataset
            )
            proposed_updates = proposal_runner(candidate, dataset_for_attempt, components_to_update)
            last_proposed_updates = {name: _raw_text(text) for name, text in proposed_updates.items()}

            if not proposed_updates or all(candidate.get(name) == text for name, text in proposed_updates.items()):
                rejection_feedback = [
                    "The proposal did not modify any target file. "
                    "Return updated file contents instead of the current text."
                ]
                continue

            proposed_candidate = _validate_candidate_keys({**candidate, **proposed_updates}, expected_keys)
            rejection_feedback = _validate_candidate_content(proposed_candidate)
            if not rejection_feedback and candidate_checker is not None:
                rejection_feedback = candidate_checker(proposed_candidate)

            if not rejection_feedback:
                return proposed_updates

            log_message = (
                f"Rejected unevaluated proposal attempt {attempt}/{max_attempts} due to validation errors: "
                + "; ".join(rejection_feedback)
            )
            if attempt < max_attempts:
                LOGGER.debug(log_message)
            else:
                LOGGER.warning(log_message)

        dump_dir = ROOT / "runs" / "research" / "_rejected_proposals"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / f"{_slugify('__'.join(components_to_update) or 'candidate')}.json"
        dump_path.write_text(
            json.dumps(
                {
                    "components_to_update": list(components_to_update),
                    "validation_errors": rejection_feedback,
                    "raw_updates": last_proposed_updates,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            "Failed to produce a valid candidate before benchmark evaluation. "
            f"Last validation errors: {rejection_feedback}"
        )

    return proposer

class _ProposalLogText(str):
    """String payload that preserves file contents while shortening GEPA log output."""

    def __new__(cls, value: str, summary: str):
        obj = super().__new__(cls, value)
        obj.summary = summary
        return obj

    def __str__(self) -> str:
        return self.summary


def _proposal_log_summary(file_path: str, previous_text: str, new_text: str) -> str:
    old_lines = previous_text.splitlines()
    new_lines = new_text.splitlines()
    changed_lines = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm=""):
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed_lines += 1

    if changed_lines == 0 and previous_text != new_text:
        changed_lines = 1

    return (
        f"[applying code modification to {Path(file_path).name}: "
        f"{changed_lines} changed lines, {len(new_lines)} total lines]"
    )


_FENCED_CODE_BLOCK_RE = re.compile(r"```[\w.+-]*\n(?P<body>.*?)\n```", re.DOTALL)
_PYTHON_CODE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"from\b|import\b|class\b|def\b|async\s+def\b|@|if\b|elif\b|else:|for\b|while\b|try:|except\b|finally:|with\b|"
    r"return\b|raise\b|assert\b|pass\b|break\b|continue\b|global\b|nonlocal\b|del\b|lambda\b|yield\b|"
    r"[A-Za-z_][\w\.]*\s*=|"
    r'"""|\'\'\'|#'
    r")"
)


def _is_valid_python_source(text: str, file_path: str) -> bool:
    try:
        compile(text, file_path, "exec")
        return True
    except SyntaxError:
        return False


def _recover_python_source(text: str, file_path: str) -> str:
    """Recover Python source from chatty model output by trimming prose wrappers."""
    stripped = text.strip()
    if not stripped:
        return stripped
    if _is_valid_python_source(stripped, file_path):
        return stripped

    lines = stripped.splitlines()
    start_candidates = [0]
    for idx, line in enumerate(lines):
        if _PYTHON_CODE_PREFIX_RE.match(line):
            start_candidates.append(idx)
            break

    seen_spans: set[tuple[int, int]] = set()
    for start in start_candidates:
        for end in range(len(lines), start, -1):
            span = (start, end)
            if span in seen_spans:
                continue
            seen_spans.add(span)
            candidate = "\n".join(lines[start:end]).strip()
            if candidate and _is_valid_python_source(candidate, file_path):
                return candidate

    return stripped


def _normalize_proposed_text(text: str, file_path: str) -> str:
    """Strip common chat wrappers from proposal output and recover source text when possible."""
    stripped = text.strip()
    matches = [match.group("body") for match in _FENCED_CODE_BLOCK_RE.finditer(stripped)]
    if matches:
        # Prefer the longest fenced block when the model mixes explanation and code samples.
        stripped = max(matches, key=len).strip()

    if Path(file_path).suffix.lower() == ".py":
        return _recover_python_source(stripped, file_path)
    return stripped


def _raw_text(value: str) -> str:
    """Return the underlying string value even for _ProposalLogText wrappers."""
    return value[:] if isinstance(value, str) else str(value)


def _build_reflection_prompt_templates(
    editable_files: Sequence[str],
    objective: str,
    background: str = "",
) -> dict[str, str]:
    templates: dict[str, str] = {}
    domain_context = background.strip()
    for file_path in editable_files:
        templates[file_path] = (
            "You are optimizing a BoxMOT source artifact.\n\n"
            f"Target file: {file_path}\n"
            f"Optimization goal: {objective}\n\n"
            "Proposal policy:\n"
            "- Prefer algorithmic tracking improvements over superficial edits.\n"
            "- Do not spend a proposal on isolated single-variable, threshold, constant, or formatting-only changes.\n"
            "- Avoid rename-only changes, import/export churn, and other non-behavioral cleanup.\n"
            "- Only change a scalar or hyperparameter when it is required as one part of a broader "
            "algorithmic modification.\n"
            "- Strong proposals usually modify the association, lifecycle, motion, or geometry logic "
            "in a coherent way.\n"
            "- Return only raw file contents. Do not wrap the response in Markdown fences or extra commentary.\n\n"
            + (
                "Domain context:\n"
                "```\n"
                f"{domain_context}\n"
                "```\n\n"
                if domain_context
                else ""
            )
            +
            (
            "Current file contents:\n"
            "```\n"
            "<curr_param>\n"
            "```\n\n"
            "Evaluation feedback:\n"
            "```\n"
            "<side_info>\n"
            "```\n\n"
            "Return the full replacement contents for the same file. "
            "Preserve public interfaces and avoid unrelated churn."
            )
        )
    return templates
