from __future__ import annotations

import argparse
from collections.abc import Callable
import difflib
import importlib
from importlib.metadata import PackageNotFoundError, distribution
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

from boxmot.configs import DEFAULT_DETECTOR, DEFAULT_REID
from boxmot.data.dataset import _collect_seq_info
from boxmot.utils import ROOT, logger as LOGGER
from boxmot.utils.benchmark_config import (
    apply_benchmark_config,
    resolve_required_reid_model,
    resolve_required_yolo_model,
)
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.compat import dataclass_slots_kwargs
from boxmot.utils.misc import resolve_model_path

RESEARCH_EXTRA = "research"
RESEARCH_METRICS = ("HOTA", "IDF1", "MOTA")
DEFAULT_PROPOSAL_MODEL = "openai/gpt-5.4"
DEFAULT_PROPOSAL_MODEL_KWARGS = {"reasoning_effort": "medium"}
_RESEARCH_ROOT = ".boxmot_research"
_PROPOSAL_VALIDATION_ATTEMPTS = 3
_PROPOSAL_API_KEY_ENV_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}
TRACKEVAL_METRIC_GLOSSARY = {
    "HOTA": "Higher is better. Overall tracking quality balancing detection and association.",
    "DetA": "Higher is better. Detection accuracy within the HOTA family.",
    "AssA": "Higher is better. Association accuracy; rewards stable identities over time.",
    "DetRe": "Higher is better. Detection recall within the HOTA family.",
    "DetPr": "Higher is better. Detection precision within the HOTA family.",
    "AssRe": "Higher is better. Association recall within the HOTA family.",
    "AssPr": "Higher is better. Association precision within the HOTA family.",
    "LocA": "Higher is better. Localization quality for matched detections.",
    "OWTA": "Higher is better. Additional HOTA-family summary reported by TrackEval.",
    "HOTA(0)": "Higher is better. HOTA at the loosest TrackEval alpha threshold.",
    "LocA(0)": "Higher is better. Localization accuracy at the loosest HOTA alpha threshold.",
    "HOTALocA(0)": "Higher is better. Product of HOTA(0) and LocA(0).",
    "MOTA": "Higher is better. CLEAR overall score combining FN, FP, and ID switches.",
    "MOTP": "Higher is better. Precision of matched-object localization in CLEAR.",
    "MODA": "Higher is better. Detection accuracy variant from CLEAR.",
    "CLR_Re": "Higher is better. CLEAR recall.",
    "CLR_Pr": "Higher is better. CLEAR precision.",
    "MTR": "Higher is better. Ratio of ground-truth trajectories that are mostly tracked.",
    "PTR": "Higher is better. Ratio of ground-truth trajectories that are partially tracked.",
    "MLR": "Lower is better. Ratio of ground-truth trajectories that are mostly lost.",
    "sMOTA": "Higher is better. Soft MOTA variant reported by TrackEval.",
    "CLR_TP": "Context count. Number of matched detections.",
    "CLR_FN": "Lower is better. Number of missed detections.",
    "CLR_FP": "Lower is better. Number of false detections.",
    "IDSW": "Lower is better. Number of identity switches.",
    "MT": "Higher is better. Count of mostly tracked ground-truth trajectories.",
    "PT": "Context count. Number of partially tracked ground-truth trajectories.",
    "ML": "Lower is better. Count of mostly lost ground-truth trajectories.",
    "Frag": "Lower is better. Number of trajectory fragmentations.",
    "IDF1": "Higher is better. Identity F1 score.",
    "IDR": "Higher is better. Identity recall.",
    "IDP": "Higher is better. Identity precision.",
    "IDTP": "Higher is better. Identity true positives.",
    "IDFN": "Lower is better. Identity false negatives.",
    "IDFP": "Lower is better. Identity false positives.",
    "Dets": "Context count. Number of tracker detections evaluated.",
    "GT_Dets": "Context count. Number of ground-truth detections.",
    "IDs": "Context count. Number of tracker identities used.",
    "GT_IDs": "Context count. Number of ground-truth identities.",
}

_EVAL_SNIPPET = r"""
import json
import sys
import traceback
from pathlib import Path

from boxmot.configs import build_mode_namespace
from boxmot.engine.evaluator import eval_setup, run_generate_dets_embs, run_generate_mot_results, run_trackeval
from boxmot.utils.evaluation.results import build_trackeval_feedback

payload = json.loads(Path(sys.argv[1]).read_text())

try:
    args = build_mode_namespace("eval", payload)
    eval_setup(args)
    run_generate_dets_embs(args)
    run_generate_mot_results(args)
    feedback = build_trackeval_feedback(run_trackeval(args, verbose=False))
    print(json.dumps({"ok": True, **feedback}, sort_keys=True))
except Exception as exc:
    print(
        json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            sort_keys=True,
        )
    )
    sys.exit(1)
"""

_PREFLIGHT_SNIPPET = r"""
import importlib
import json
import sys
import traceback
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())

try:
    importlib.invalidate_caches()
    for module_name in payload.get("modules", []):
        importlib.import_module(module_name)
    print(json.dumps({"ok": True}, sort_keys=True))
except Exception as exc:
    print(
        json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            sort_keys=True,
        )
    )
    sys.exit(1)
"""


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


def _aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in RESEARCH_METRICS}
    count = float(len(rows))
    return {
        metric: sum(float(row.get(metric, 0.0)) for row in rows) / count
        for metric in RESEARCH_METRICS
    }


def _count_changed_lines(before: str, after: str) -> tuple[int, int]:
    diff = difflib.ndiff(before.splitlines(), after.splitlines())
    added = sum(1 for line in diff if line.startswith("+ "))
    removed = sum(1 for line in diff if line.startswith("- "))
    return added, removed


def _metric_delta(
    current: Mapping[str, int | float],
    baseline: Mapping[str, int | float] | None,
) -> dict[str, float]:
    if not baseline:
        return {}

    delta: dict[str, float] = {}
    for key, value in current.items():
        baseline_value = baseline.get(key)
        if isinstance(value, bool) or isinstance(baseline_value, bool):
            continue
        if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta[key] = float(value) - float(baseline_value)
    return delta


def _nested_metric_delta(
    current: Mapping[str, Mapping[str, int | float]],
    baseline: Mapping[str, Mapping[str, int | float]] | None,
) -> dict[str, dict[str, float]]:
    if not baseline:
        return {}

    deltas: dict[str, dict[str, float]] = {}
    for name, metrics in current.items():
        metric_delta = _metric_delta(metrics, baseline.get(name, {}))
        if metric_delta:
            deltas[name] = metric_delta
    return deltas


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

    src_path = Path(src).resolve()
    if src_path == (ROOT / "boxmot" / "engine" / "trackeval").resolve() and "data" in names:
        ignored.add("data")
    return ignored


def _resolve_benchmark_runtime(
    benchmark: str | Path,
    *,
    source: str | Path | None = None,
    detector: str | Path | None = None,
    reid: str | Path | None = None,
) -> tuple[Path, str, Path, Path, dict[str, Any]]:
    probe = SimpleNamespace(data=str(benchmark))
    cfg = apply_benchmark_config(probe)
    if cfg is None:
        raise FileNotFoundError(f"Unable to resolve benchmark config: {benchmark}")

    benchmark_id = str(getattr(probe, "benchmark_id", getattr(probe, "benchmark", benchmark)))
    source_root = Path(source or probe.source).resolve()
    detector_ref = detector or resolve_required_yolo_model(cfg) or DEFAULT_DETECTOR
    reid_ref = reid or resolve_required_reid_model(cfg) or DEFAULT_REID

    return (
        source_root,
        benchmark_id,
        resolve_model_path(detector_ref).resolve(),
        resolve_model_path(reid_ref).resolve(),
        cfg,
    )


def _discover_sequences(source_root: Path) -> list[dict[str, str]]:
    seq_paths, _ = _collect_seq_info(source_root)
    examples: list[dict[str, str]] = []
    for img_dir in seq_paths:
        seq_dir = img_dir.parent if img_dir.name == "img1" else img_dir
        examples.append(
            {
                "sequence": seq_dir.name,
                "sequence_dir": str(seq_dir.resolve()),
            }
        )
    if not examples:
        raise ValueError(f"No benchmark sequences found under {source_root}")
    return examples


def _split_examples(
    examples: Sequence[Mapping[str, str]],
    *,
    validation_split: float,
    train_sequences: Sequence[str] | None = None,
    val_sequences: Sequence[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_name = {str(example["sequence"]): dict(example) for example in examples}

    if train_sequences or val_sequences:
        missing = [name for name in [*(train_sequences or ()), *(val_sequences or ())] if name not in by_name]
        if missing:
            raise ValueError(f"Unknown sequence(s): {missing}")
        train = [by_name[name] for name in train_sequences or ()]
        val = [by_name[name] for name in val_sequences or ()]
        if not train:
            raise ValueError("train_sequences resolved to an empty set")
        return train, val

    if len(examples) <= 1 or validation_split <= 0:
        return [dict(example) for example in examples], []

    val_count = max(1, int(round(len(examples) * validation_split)))
    if val_count >= len(examples):
        val_count = len(examples) - 1

    train = [dict(example) for example in examples[:-val_count]]
    val = [dict(example) for example in examples[-val_count:]]
    return train, val


def _select_examples(
    examples: Sequence[Mapping[str, str]],
    *,
    train_sequences: Sequence[str] | None = None,
    val_sequences: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    by_name = {str(example["sequence"]): dict(example) for example in examples}
    requested = [*(train_sequences or ()), *(val_sequences or ())]
    if not requested:
        return [dict(example) for example in examples]

    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown sequence(s): {missing}")

    ordered_unique = list(dict.fromkeys(requested))
    return [by_name[name] for name in ordered_unique]


def _normalize_editable_files(
    tracker: str,
    editable_files: Sequence[str | Path] | None,
) -> tuple[str, ...]:
    if editable_files:
        files = [Path(path) for path in editable_files]
    else:
        tracker_dir = ROOT / "boxmot" / "trackers" / tracker.lower()
        if not tracker_dir.exists():
            raise FileNotFoundError(f"Tracker source directory not found: {tracker_dir}")

        main_impl = tracker_dir / f"{tracker.lower()}.py"
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
                    "The proposal did not modify any target file. Return updated file contents instead of the current text."
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


def _ensure_not_local_gepa_path(path: Path | None) -> None:
    if path is None:
        raise RuntimeError("Unable to determine the installed gepa package path")
    local_checkout = (ROOT / "gepa").resolve()
    resolved = path.resolve()
    if resolved == local_checkout or _is_relative_to(resolved, local_checkout):
        raise RuntimeError(
            "The resolved `gepa` package points at the local `./gepa` checkout. "
            "Install the published pip package instead."
        )


def _import_installed_gepa() -> Any:
    try:
        gepa_dist = distribution("gepa")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "`gepa` is not installed. Install the BoxMOT research extra with "
            "`uv sync --extra research` or `uv pip install '.[research]'`."
        ) from exc

    dist_root = Path(gepa_dist.locate_file("")).resolve()
    _ensure_not_local_gepa_path(dist_root)

    if str(dist_root) not in sys.path:
        sys.path.insert(0, str(dist_root))

    module = importlib.import_module("gepa")
    module_file = getattr(module, "__file__", None)
    if module_file:
        _ensure_not_local_gepa_path(Path(module_file))
    return module


def _load_gepa_litellm_factory() -> Callable[[str], Any] | None:
    """Resolve the published GEPA liteLLM factory when that layout is installed."""
    try:
        module = importlib.import_module("gepa.optimize_anything")
    except ImportError:
        return None

    factory = getattr(module, "make_litellm_lm", None)
    return factory if callable(factory) else None


def _resolve_proposal_api_key_env(model_name: str, configured_env: str | None) -> str | None:
    """Resolve the provider API-key environment variable for a proposal model."""
    if configured_env:
        return str(configured_env).strip() or None

    provider = str(model_name).split("/", 1)[0].strip().lower()
    return _PROPOSAL_API_KEY_ENV_BY_PROVIDER.get(provider)


def _prepare_proposal_model_env(model_name: str, model_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply proposal-model credential settings and return kwargs safe for LM construction."""
    sanitized_kwargs = dict(model_kwargs)
    api_key = sanitized_kwargs.pop("api_key", None)
    api_key_env = sanitized_kwargs.pop("api_key_env", None)

    if api_key is None:
        return sanitized_kwargs

    env_name = _resolve_proposal_api_key_env(model_name, api_key_env)
    if env_name is None:
        raise ValueError(
            f"Cannot infer an API-key environment variable for proposal model '{model_name}'. "
            "Pass --proposal-api-key-env ENV_NAME alongside --proposal-api-key."
        )

    os.environ[env_name] = str(api_key)
    return sanitized_kwargs


def _build_reflection_lm(model_name: str, model_kwargs: Mapping[str, Any]) -> Any:
    """Construct a GEPA-compatible reflection LM across published package layouts."""
    sanitized_kwargs = _prepare_proposal_model_env(model_name, model_kwargs)
    make_litellm_lm = _load_gepa_litellm_factory()
    if make_litellm_lm is not None:
        return make_litellm_lm(model_name)

    from gepa.lm import LM

    return LM(model_name, **sanitized_kwargs)


def _run_instruction_proposal_signature(
    signature_cls: Any,
    *,
    lm: Any,
    input_dict: Mapping[str, Any],
) -> dict[str, str]:
    """Run a GEPA instruction signature across published/newer API variants."""
    runner = getattr(signature_cls, "run_with_metadata", None)
    if callable(runner):
        result, _, _ = runner(lm=lm, input_dict=input_dict)
        return result

    runner = getattr(signature_cls, "run", None)
    if callable(runner):
        return runner(lm=lm, input_dict=input_dict)

    raise AttributeError(f"{signature_cls!r} does not expose a supported run method")


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
            "- Only change a scalar or hyperparameter when it is required as one part of a broader algorithmic modification.\n"
            "- Strong proposals usually modify the association, lifecycle, motion, or geometry logic in a coherent way.\n"
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


@dataclass(frozen=True, **dataclass_slots_kwargs())
class RegressionPenalties:
    idf1_penalty: float = 1.0
    mota_penalty: float = 1.0
    idf1_tolerance: float = 0.0
    mota_tolerance: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("idf1_penalty", "mota_penalty", "idf1_tolerance", "mota_tolerance"):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "idf1_penalty": float(self.idf1_penalty),
            "mota_penalty": float(self.mota_penalty),
            "idf1_tolerance": float(self.idf1_tolerance),
            "mota_tolerance": float(self.mota_tolerance),
        }


@dataclass(frozen=True, **dataclass_slots_kwargs())
class ResearchConfig:
    tracker: str
    benchmark: str
    source: Path | None = None
    detector: Path | None = None
    reid: Path | None = None
    editable_files: tuple[str, ...] | None = None
    extra_context_files: tuple[str, ...] = ()
    train_sequences: tuple[str, ...] | None = None
    val_sequences: tuple[str, ...] | None = None
    validation_split: float = 0.25
    proposal_model: str = DEFAULT_PROPOSAL_MODEL
    proposal_model_kwargs: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PROPOSAL_MODEL_KWARGS))
    penalties: RegressionPenalties = field(default_factory=RegressionPenalties)
    max_metric_calls: int = 24
    reflection_minibatch_size: int = 2
    eval_timeout: float = 60.0
    project: Path = Path("runs")
    name: str = "research"
    progress_bar: bool = True
    keep_workspace: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> ResearchConfig:
        benchmark = getattr(args, "benchmark", None) or getattr(args, "data", "")
        detector = None
        if getattr(args, "detector_explicit", False) and getattr(args, "detector", None):
            detector = Path(args.detector[0])

        reid = None
        if getattr(args, "reid_explicit", False) and getattr(args, "reid", None):
            reid = Path(args.reid[0])

        proposal_model_kwargs = dict(getattr(args, "proposal_model_kwargs", DEFAULT_PROPOSAL_MODEL_KWARGS) or {})
        if "reasoning_effort" not in proposal_model_kwargs:
            proposal_model_kwargs["reasoning_effort"] = DEFAULT_PROPOSAL_MODEL_KWARGS["reasoning_effort"]

        proposal_api_key = getattr(args, "proposal_api_key", None)
        if proposal_api_key:
            proposal_model_kwargs["api_key"] = str(proposal_api_key)

        proposal_api_key_env = getattr(args, "proposal_api_key_env", None)
        if proposal_api_key_env:
            proposal_model_kwargs["api_key_env"] = str(proposal_api_key_env)

        return cls(
            tracker=str(getattr(args, "tracker", "")),
            benchmark=str(benchmark),
            source=Path(getattr(args, "source")) if getattr(args, "source", None) else None,
            detector=detector,
            reid=reid,
            proposal_model=str(getattr(args, "proposal_model", DEFAULT_PROPOSAL_MODEL)),
            proposal_model_kwargs=proposal_model_kwargs,
            penalties=RegressionPenalties(
                idf1_penalty=float(getattr(args, "idf1_penalty", 1.0)),
                mota_penalty=float(getattr(args, "mota_penalty", 1.0)),
                idf1_tolerance=float(getattr(args, "idf1_tolerance", 0.0)),
                mota_tolerance=float(getattr(args, "mota_tolerance", 0.0)),
            ),
            max_metric_calls=int(getattr(args, "max_metric_calls", 24)),
            reflection_minibatch_size=int(getattr(args, "reflection_minibatch_size", 2)),
            eval_timeout=float(getattr(args, "eval_timeout", 900.0)),
            project=Path(getattr(args, "project", "runs")),
            name=str(getattr(args, "name", "research")),
            keep_workspace=bool(getattr(args, "keep_workspace", False)),
        )


@dataclass(**dataclass_slots_kwargs())
class ResearchResult:
    tracker: str
    benchmark: str
    proposal_model: str
    run_dir: Path
    best_candidate_dir: Path
    editable_files: tuple[str, ...]
    train_sequences: tuple[str, ...]
    val_sequences: tuple[str, ...]
    baseline_summary: dict[str, int | float]
    best_summary: dict[str, int | float]
    delta_summary: dict[str, float]
    workspace_dir: Path | None = None

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        def _format_metrics(metrics: Mapping[str, int | float], *, signed: bool = False) -> str:
            parts = []
            for metric in RESEARCH_METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    fmt = f"{float(value):+.3f}" if signed else f"{float(value):.3f}"
                    parts.append(f"{metric}={fmt}")
            return " ".join(parts) if parts else "n/a"

        lines = [
            "RESEARCH SUMMARY",
            f"Tracker: {self.tracker}",
            f"Benchmark: {self.benchmark}",
            f"Proposal model: {self.proposal_model}",
            f"Baseline: {_format_metrics(self.baseline_summary)}",
            f"Best: {_format_metrics(self.best_summary)}",
            f"Delta: {_format_metrics(self.delta_summary, signed=True)}",
            f"Best candidate dir: {self.best_candidate_dir}",
        ]
        if self.workspace_dir is not None:
            lines.append(f"Workspace: {self.workspace_dir}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.render())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker,
            "benchmark": self.benchmark,
            "proposal_model": self.proposal_model,
            "run_dir": str(self.run_dir),
            "best_candidate_dir": str(self.best_candidate_dir),
            "editable_files": list(self.editable_files),
            "train_sequences": list(self.train_sequences),
            "val_sequences": list(self.val_sequences),
            "baseline_summary": self.baseline_summary,
            "best_summary": self.best_summary,
            "delta_summary": self.delta_summary,
            "workspace_dir": None if self.workspace_dir is None else str(self.workspace_dir),
        }


class TrackerResearcher:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.cache_project_dir = config.project.resolve()
        self.run_dir = (config.project / "research" / f"{_slugify(config.tracker)}_{_slugify(config.benchmark)}").resolve()
        self.boxmot_project_dir = self.run_dir / "boxmot_runs"
        self.gepa_run_dir = self.run_dir / "gepa"
        self.workspace_dir: Path | None = None

        (
            self.source_root,
            self.benchmark_id,
            self.detector_path,
            self.reid_path,
            self.benchmark_cfg,
        ) = _resolve_benchmark_runtime(
            config.benchmark,
            source=config.source,
            detector=config.detector,
            reid=config.reid,
        )

        self.editable_files = _normalize_editable_files(config.tracker, config.editable_files)
        self.seed_candidate = _read_candidate(self.editable_files)

        all_examples = _discover_sequences(self.source_root)
        self.all_examples = _select_examples(
            all_examples,
            train_sequences=config.train_sequences,
            val_sequences=config.val_sequences,
        )
        self.selected_sequences = tuple(example["sequence"] for example in self.all_examples)
        self.train_examples = [dict(example) for example in self.all_examples]
        self.val_examples = [dict(example) for example in self.all_examples]
        self.penalties = config.penalties
        self.baseline_summary: dict[str, int | float] | None = None
        self.baseline_summary_label: str = ""
        self.baseline_per_sequence_metrics: dict[str, dict[str, int | float]] = {}
        self.baseline_per_class_metrics: dict[str, dict[str, int | float]] = {}

    def _ensure_dependencies(self) -> None:
        RequirementsChecker().sync_extra(RESEARCH_EXTRA, verbose=False)
        _import_installed_gepa()

    def _reset_gepa_run_dir(self) -> None:
        if self.gepa_run_dir.exists():
            LOGGER.info(f"Resetting stale GEPA state in {self.gepa_run_dir}")
            shutil.rmtree(self.gepa_run_dir, ignore_errors=True)
        self.gepa_run_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_workspace(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.boxmot_project_dir.mkdir(parents=True, exist_ok=True)
        self._reset_gepa_run_dir()

        workspace = Path(
            tempfile.mkdtemp(prefix="workspace_", dir=str(self.run_dir))
        )
        shutil.copytree(ROOT, workspace, dirs_exist_ok=True, ignore=_workspace_copy_ignore)
        self.workspace_dir = workspace
        return workspace

    def _write_candidate_to_workspace(self, candidate: Mapping[str, str]) -> None:
        if self.workspace_dir is None:
            raise RuntimeError("Workspace has not been prepared")
        for file_path, content in candidate.items():
            dst = self.workspace_dir / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")

    def _subset_source_root(self, sequence_names: Sequence[str]) -> Path:
        if self.workspace_dir is None:
            raise RuntimeError("Workspace has not been prepared")

        subset_key = "__".join(sorted(sequence_names))
        container = self.workspace_dir / _RESEARCH_ROOT / "subsets" / _slugify(subset_key)
        data_root = container / "data"
        ann_root = container / "annotations"

        if data_root.exists():
            return data_root

        data_root.mkdir(parents=True, exist_ok=True)
        source_annotations = self.source_root.parent / "annotations"

        by_name = {example["sequence"]: Path(example["sequence_dir"]) for example in self.all_examples}
        for sequence_name in sequence_names:
            seq_dir = by_name[sequence_name]
            target = data_root / sequence_name
            if not target.exists():
                try:
                    target.symlink_to(seq_dir, target_is_directory=True)
                except OSError:
                    shutil.copytree(seq_dir, target, dirs_exist_ok=True)

            if source_annotations.exists():
                ann_file = source_annotations / f"{sequence_name}.txt"
                if ann_file.exists():
                    ann_root.mkdir(parents=True, exist_ok=True)
                    ann_target = ann_root / ann_file.name
                    if not ann_target.exists():
                        try:
                            ann_target.symlink_to(ann_file)
                        except OSError:
                            shutil.copy2(ann_file, ann_target)

        return data_root

    def _score_candidate(self, metrics: Mapping[str, int | float]) -> tuple[float, dict[str, float]]:
        if self.baseline_summary is None:
            raise RuntimeError("Baseline summary is not available for candidate scoring")

        hota = float(metrics.get("HOTA", 0.0))
        baseline_idf1 = float(self.baseline_summary.get("IDF1", 0.0))
        baseline_mota = float(self.baseline_summary.get("MOTA", 0.0))
        idf1 = float(metrics.get("IDF1", 0.0))
        mota = float(metrics.get("MOTA", 0.0))

        idf1_regression = max(0.0, baseline_idf1 - idf1 - float(self.penalties.idf1_tolerance))
        mota_regression = max(0.0, baseline_mota - mota - float(self.penalties.mota_tolerance))
        total_penalty = (
            idf1_regression * float(self.penalties.idf1_penalty)
            + mota_regression * float(self.penalties.mota_penalty)
        )
        score = hota - total_penalty

        return score, {
            "HOTA": hota,
            "baseline_IDF1": baseline_idf1,
            "baseline_MOTA": baseline_mota,
            "idf1_regression": idf1_regression,
            "mota_regression": mota_regression,
            "idf1_tolerance": float(self.penalties.idf1_tolerance),
            "mota_tolerance": float(self.penalties.mota_tolerance),
            "idf1_penalty": float(self.penalties.idf1_penalty),
            "mota_penalty": float(self.penalties.mota_penalty),
            "total_penalty": total_penalty,
            "score": score,
        }

    def _build_eval_payload(self, source_root: Path, tag: str) -> dict[str, Any]:
        show_progress = bool(getattr(self.config, "progress_bar", True))
        return {
            "data": None,
            "source": str(source_root),
            "benchmark": self.benchmark_id,
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.benchmark_id,
            "tracker": self.config.tracker,
            "detector": [self.detector_path],
            "reid": [self.reid_path],
            "project": self.boxmot_project_dir,
            "cache_project": self.cache_project_dir,
            "name": tag,
            "verbose": False,
            "show": False,
            "show_progress": show_progress,
            "save": False,
            "save_txt": False,
            "save_crop": False,
            "ci": False,
            "detector_explicit": True,
            "reid_explicit": True,
        }

    def _run_eval_subprocess(self, manifest_path: Path) -> dict[str, Any]:
        stderr_target = None if bool(getattr(self.config, "progress_bar", False)) else subprocess.PIPE
        proc = subprocess.Popen(
            [sys.executable, "-c", _EVAL_SNIPPET, str(manifest_path)],
            cwd=self.workspace_dir,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=self.config.eval_timeout)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _terminate_subprocess_tree(proc, graceful=False)
            return {
                "ok": False,
                "summary": {metric: 0.0 for metric in RESEARCH_METRICS},
                "summary_label": "",
                "per_sequence_metrics": {},
                "per_class_metrics": {},
                "stdout": stdout or exc.stdout or "",
                "stderr": stderr or exc.stderr or "",
                "error": f"Evaluation timed out after {self.config.eval_timeout:.1f}s",
                "traceback": "",
                "static_errors": [],
            }
        except KeyboardInterrupt:
            _terminate_subprocess_tree(proc, graceful=True)
            raise

        try:
            payload_out = _parse_last_json_line(stdout)
        except Exception as exc:
            payload_out = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        ok = bool(proc.returncode == 0 and payload_out.get("ok"))
        stderr_text = stderr or ""
        summary = {metric: 0.0 for metric in RESEARCH_METRICS}
        summary_payload = payload_out.get("summary", {})
        if isinstance(summary_payload, dict):
            for key, value in summary_payload.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    summary[str(key)] = value

        return {
            "ok": ok,
            "summary": summary,
            "summary_label": str(payload_out.get("summary_label", "")),
            "per_sequence_metrics": (
                payload_out.get("per_sequence_metrics", {})
                if isinstance(payload_out.get("per_sequence_metrics", {}), dict)
                else {}
            ),
            "per_class_metrics": (
                payload_out.get("per_class_metrics", {})
                if isinstance(payload_out.get("per_class_metrics", {}), dict)
                else {}
            ),
            "stdout": stdout,
            "stderr": stderr_text,
            "error": None if ok else payload_out.get("error", "Evaluation subprocess failed"),
            "traceback": payload_out.get("traceback", ""),
            "static_errors": [],
        }

    def _run_preflight_subprocess(self, manifest_path: Path) -> dict[str, Any]:
        proc = subprocess.Popen(
            [sys.executable, "-c", _PREFLIGHT_SNIPPET, str(manifest_path)],
            cwd=self.workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        timeout = min(float(self.config.eval_timeout), 30.0)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _terminate_subprocess_tree(proc, graceful=False)
            return {
                "ok": False,
                "error": f"Preflight import timed out after {timeout:.1f}s",
                "traceback": "",
                "stdout": stdout or exc.stdout or "",
                "stderr": stderr or exc.stderr or "",
            }
        except KeyboardInterrupt:
            _terminate_subprocess_tree(proc, graceful=True)
            raise

        try:
            payload_out = _parse_last_json_line(stdout)
        except Exception:
            payload_out = {
                "ok": False,
                "error": "No JSON payload found in preflight subprocess stdout",
                "traceback": traceback.format_exc(),
            }

        ok = bool(proc.returncode == 0 and payload_out.get("ok"))
        return {
            "ok": ok,
            "error": None if ok else payload_out.get("error", "Preflight subprocess failed"),
            "traceback": payload_out.get("traceback", ""),
            "stdout": stdout,
            "stderr": stderr,
        }

    def _preflight_candidate(self, candidate: Mapping[str, str], tag: str) -> list[str]:
        validated = _validate_candidate_keys(candidate, self.editable_files)
        static_errors = _validate_candidate_content(validated)
        if static_errors:
            return static_errors

        self._write_candidate_to_workspace(validated)
        modules = _candidate_import_modules(self.editable_files)
        if not modules:
            return []

        manifest_dir = self.workspace_dir / _RESEARCH_ROOT / "preflight"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{_slugify(tag)}.json"
        manifest_path.write_text(json.dumps({"modules": list(modules)}, indent=2), encoding="utf-8")

        result = self._run_preflight_subprocess(manifest_path)
        if result["ok"]:
            return []

        errors = [str(result["error"])]
        if result.get("traceback"):
            errors.append(str(result["traceback"]))
        return errors

    def _run_candidate_eval(self, candidate: Mapping[str, str], sequence_names: Sequence[str], tag: str) -> dict[str, Any]:
        validated = _validate_candidate_keys(candidate, self.editable_files)
        static_errors = _validate_candidate_content(validated)
        if static_errors:
            return {
                "ok": False,
                "error": "Static validation failed",
                "static_errors": static_errors,
                "summary": {metric: 0.0 for metric in RESEARCH_METRICS},
                "summary_label": "",
                "per_sequence_metrics": {},
                "per_class_metrics": {},
                "stdout": "",
                "stderr": "",
            }

        self._write_candidate_to_workspace(validated)
        subset_root = self._subset_source_root(sequence_names)
        payload = self._build_eval_payload(subset_root, tag)

        manifest_dir = self.workspace_dir / _RESEARCH_ROOT / "payloads"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{_slugify(tag)}.json"
        manifest_path.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")
        return self._run_eval_subprocess(manifest_path)

    def _combined_benchmark_eval(self, candidate: Mapping[str, str], tag: str) -> dict[str, Any]:
        result = self._run_candidate_eval(
            candidate,
            self.selected_sequences,
            tag=tag,
        )
        if not result["ok"]:
            raise RuntimeError(f"Combined benchmark evaluation failed for {tag}: {result['error']}")
        return result

    def _combined_benchmark_summary(self, candidate: Mapping[str, str], tag: str) -> dict[str, int | float]:
        return self._combined_benchmark_eval(candidate, tag)["summary"]

    def _candidate_evaluator(self, candidate: Mapping[str, str]) -> tuple[float, dict[str, Any]]:
        validated = _validate_candidate_keys(candidate, self.editable_files)
        result = self._run_candidate_eval(
            validated,
            self.selected_sequences,
            tag="candidate_all_sequences",
        )
        change_summary = _candidate_change_summary(self.seed_candidate, validated)

        if not result["ok"]:
            side_info = {
                "scores": {"score": 0.0},
                "TrackEval Summary Label": result.get("summary_label", self.baseline_summary_label),
                "Sequences": list(self.selected_sequences),
                "Combined Metrics": result["summary"],
                "Baseline Metrics": self.baseline_summary or {metric: 0.0 for metric in RESEARCH_METRICS},
                "Combined Delta vs Baseline": _metric_delta(
                    result["summary"],
                    self.baseline_summary or {},
                ),
                "Per-Sequence Metrics": result.get("per_sequence_metrics", {}),
                "Per-Sequence Delta vs Baseline": _nested_metric_delta(
                    result.get("per_sequence_metrics", {}),
                    self.baseline_per_sequence_metrics,
                ),
                "Per-Class Combined Metrics": result.get("per_class_metrics", {}),
                "Per-Class Delta vs Baseline": _nested_metric_delta(
                    result.get("per_class_metrics", {}),
                    self.baseline_per_class_metrics,
                ),
                "Metric Glossary": TRACKEVAL_METRIC_GLOSSARY,
                "Changed Files": change_summary,
                "Error": result["error"],
                "Validation": result.get("static_errors", []),
                "Traceback": result.get("traceback", ""),
                "Stdout": result.get("stdout", ""),
                "Stderr": result.get("stderr", ""),
            }
            return 0.0, side_info

        summary = result["summary"]
        score, penalty_breakdown = self._score_candidate(summary)

        side_info = {
            "scores": {"score": score},
            "TrackEval Summary Label": result.get("summary_label", self.baseline_summary_label),
            "Sequences": list(self.selected_sequences),
            "Combined Metrics": summary,
            "Baseline Metrics": self.baseline_summary,
            "Combined Delta vs Baseline": _metric_delta(summary, self.baseline_summary or {}),
            "Per-Sequence Metrics": result.get("per_sequence_metrics", {}),
            "Per-Sequence Delta vs Baseline": _nested_metric_delta(
                result.get("per_sequence_metrics", {}),
                self.baseline_per_sequence_metrics,
            ),
            "Per-Class Combined Metrics": result.get("per_class_metrics", {}),
            "Per-Class Delta vs Baseline": _nested_metric_delta(
                result.get("per_class_metrics", {}),
                self.baseline_per_class_metrics,
            ),
            "Metric Glossary": TRACKEVAL_METRIC_GLOSSARY,
            "Penalty Breakdown": penalty_breakdown,
            "Changed Files": change_summary,
            "Stdout": result.get("stdout", ""),
            "Stderr": result.get("stderr", ""),
        }
        return score, side_info

    def _objective(self, baseline_summary: Mapping[str, int | float]) -> str:
        baseline = ", ".join(f"{metric}={baseline_summary.get(metric, 0.0):.2f}" for metric in RESEARCH_METRICS)
        return (
            f"Improve the BoxMOT tracker `{self.config.tracker}` on benchmark `{self.benchmark_id}` while "
            "preserving existing public behavior and file interfaces. Optimize the combined benchmark HOTA "
            "directly, while penalizing regressions in combined IDF1 and MOTA relative to the baseline benchmark "
            f"run. Current combined benchmark baseline: {baseline}."
        )

    def _background(self) -> str:
        editable = "\n".join(f"- {path}" for path in self.editable_files)
        selected_sequences = ", ".join(self.selected_sequences)
        extra_sections: list[str] = []

        for file_path in self.config.extra_context_files:
            abs_path = (ROOT / file_path).resolve()
            if not abs_path.exists():
                continue
            rel_path = abs_path.relative_to(ROOT).as_posix()
            extra_sections.append(
                f"\nContext file: {rel_path}\n```\n{abs_path.read_text(encoding='utf-8')}\n```"
            )

        return (
            f"Benchmark source root: {self.source_root}\n"
            f"Detector: {self.detector_path}\n"
            f"ReID: {self.reid_path}\n"
            "Editable files:\n"
            f"{editable}\n"
            f"Selected benchmark sequences: {selected_sequences}\n"
            "Default research scope is code-first: prioritize tracker implementation changes over standalone config tuning.\n"
            "Use the existing BoxMOT evaluation pipeline: generate detections/embeddings, run the tracker, "
            "then score with TrackEval over the full selected benchmark sequence set on every candidate "
            "evaluation. The scalar search score is combined HOTA minus penalties for any combined IDF1 or "
            "MOTA regression versus the baseline benchmark run. Keep imports minimal, preserve file paths, and "
            "avoid introducing new dependencies."
            + "".join(extra_sections)
        )

    def _save_candidate_snapshot(self, candidate: Mapping[str, str], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for file_path, content in candidate.items():
            dst = destination / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")

    def run(self) -> ResearchResult:
        self._ensure_dependencies()
        _import_installed_gepa()
        workspace = self._prepare_workspace()

        LOGGER.info(
            f"Starting tracker research for {self.config.tracker} on {self.benchmark_id} "
            f"with {len(self.selected_sequences)} benchmark sequence(s)"
        )
        LOGGER.info(f"Editable files: {', '.join(self.editable_files)}")

        LOGGER.info("Running baseline benchmark evaluation before GEPA search...")
        baseline_eval = self._combined_benchmark_eval(self.seed_candidate, "baseline_all_sequences")
        baseline_summary = baseline_eval["summary"]
        self.baseline_summary = baseline_summary
        self.baseline_summary_label = str(baseline_eval.get("summary_label", ""))
        self.baseline_per_sequence_metrics = dict(baseline_eval.get("per_sequence_metrics", {}))
        self.baseline_per_class_metrics = dict(baseline_eval.get("per_class_metrics", {}))
        objective = self._objective(baseline_summary)
        background = self._background()

        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        reflection_lm = _build_reflection_lm(
            self.config.proposal_model,
            self.config.proposal_model_kwargs,
        )
        reflection_templates = _build_reflection_prompt_templates(
            self.editable_files,
            objective,
            background,
        )

        def proposal_runner(
            candidate: dict[str, str],
            reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            new_texts: dict[str, str] = {}
            for name in components_to_update:
                if name not in reflective_dataset or not reflective_dataset.get(name):
                    continue
                result = _run_instruction_proposal_signature(
                    InstructionProposalSignature,
                    lm=reflection_lm,
                    input_dict={
                        "current_instruction_doc": candidate[name],
                        "dataset_with_feedback": reflective_dataset[name],
                        "prompt_template": reflection_templates.get(name),
                    },
                )
                new_instruction = _normalize_proposed_text(result["new_instruction"], name)
                new_texts[name] = _ProposalLogText(
                    new_instruction,
                    _proposal_log_summary(name, candidate[name], new_instruction),
                )
            return new_texts

        checked_candidate_proposer = _make_checked_candidate_proposer(
            proposal_runner,
            expected_keys=self.editable_files,
            candidate_checker=lambda candidate: self._preflight_candidate(candidate, "proposal_preflight"),
        )

        gepa_config = GEPAConfig(
            engine=EngineConfig(
                run_dir=str(self.gepa_run_dir),
                frontier_type="instance",
                max_metric_calls=self.config.max_metric_calls,
                display_progress_bar=self.config.progress_bar,
                parallel=False,
                max_workers=1,
                cache_evaluation=True,
                cache_evaluation_storage="disk",
                raise_on_exception=False,
                capture_stdio=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=1,
                reflection_prompt_template=reflection_templates,
                custom_candidate_proposer=checked_candidate_proposer,
            ),
            refiner=None,
            merge=None,
        )

        LOGGER.info("Baseline complete. Starting GEPA optimization...")
        result = optimize_anything(
            seed_candidate=self.seed_candidate,
            evaluator=self._candidate_evaluator,
            config=gepa_config,
        )

        best_candidate = _validate_candidate_keys(result.best_candidate, self.editable_files)
        best_eval = self._combined_benchmark_eval(best_candidate, "best_all_sequences")
        best_summary = best_eval["summary"]
        baseline_score, baseline_penalty_breakdown = self._score_candidate(baseline_summary)
        best_score, best_penalty_breakdown = self._score_candidate(best_summary)
        delta_summary = {
            metric: float(best_summary.get(metric, 0.0) - baseline_summary.get(metric, 0.0))
            for metric in RESEARCH_METRICS
        }

        best_candidate_dir = self.run_dir / "best_candidate"
        self._save_candidate_snapshot(best_candidate, best_candidate_dir)

        summary_path = self.run_dir / "research_result.json"
        summary_path.write_text(
            json.dumps(
                {
                    "tracker": self.config.tracker,
                    "benchmark": self.benchmark_id,
                    "proposal_model": self.config.proposal_model,
                    "scoring": {
                        "primary_metric": "HOTA",
                        **self.penalties.to_dict(),
                    },
                    "search_signal": "combined_benchmark",
                    "eval_timeout": self.config.eval_timeout,
                    "editable_files": list(self.editable_files),
                    "train_sequences": [example["sequence"] for example in self.train_examples],
                    "val_sequences": [example["sequence"] for example in self.val_examples],
                    "baseline_summary": baseline_summary,
                    "baseline_score": baseline_score,
                    "baseline_penalty_breakdown": baseline_penalty_breakdown,
                    "best_summary": best_summary,
                    "best_score": best_score,
                    "best_penalty_breakdown": best_penalty_breakdown,
                    "delta_summary": delta_summary,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

        workspace_result = workspace if self.config.keep_workspace else None
        if not self.config.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
            self.workspace_dir = None

        return ResearchResult(
            tracker=self.config.tracker,
            benchmark=self.benchmark_id,
            proposal_model=self.config.proposal_model,
            run_dir=self.run_dir,
            best_candidate_dir=best_candidate_dir,
            editable_files=self.editable_files,
            train_sequences=tuple(example["sequence"] for example in self.train_examples),
            val_sequences=tuple(example["sequence"] for example in self.val_examples),
            baseline_summary=baseline_summary,
            best_summary=best_summary,
            delta_summary=delta_summary,
            workspace_dir=workspace_result,
        )


def main(args: argparse.Namespace) -> ResearchResult:
    return run_research(args)


def run_research(args: argparse.Namespace) -> ResearchResult:
    config = args if isinstance(args, ResearchConfig) else ResearchConfig.from_namespace(args)
    return TrackerResearcher(config).run()


__all__ = [
    "DEFAULT_PROPOSAL_MODEL",
    "RESEARCH_METRICS",
    "RegressionPenalties",
    "ResearchConfig",
    "ResearchResult",
    "TrackerResearcher",
    "main",
    "run_research",
]
