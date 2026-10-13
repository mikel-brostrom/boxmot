"""Read saved tuning evidence and resolve the parameters actually searched."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from boxmot.engine.tuning.conditions import compile_association_schema, resolve_entry
from boxmot.engine.tuning.search_space import conditional_yaml_tree, normalize_trial_config

LOGGER = logging.getLogger(__name__)
NEAR_BEST_DELTA = 0.2
_METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "AssRe", "IDSW")


def near_best_mask(scores: pd.Series) -> pd.Series:
    """Select the inclusive HOTA band with tolerance for decimal roundoff."""
    return scores >= scores.max() - NEAR_BEST_DELTA - 1e-9


@dataclass
class ParameterSamples:
    """Finite observations of one active search dimension, retaining trial indices."""

    name: str
    kind: str
    samples: pd.DataFrame
    bounds: tuple[float, float] | None = None
    scale: str = "linear"
    options: tuple[Any, ...] = ()
    branch_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    branch_options: dict[str, tuple[Any, ...]] = field(default_factory=dict)


@dataclass
class TrialAnalysis:
    """Comparable completed trials and the metadata supporting their diagnostics."""

    trials: pd.DataFrame
    parameters: dict[str, ParameterSamples]
    baseline_index: int
    baseline_label: str
    best_index: int
    near_best: pd.Series
    sequence_hota: pd.DataFrame
    notes: list[str]
    chronological: bool
    schema: dict = field(default_factory=dict)


def _present(value: Any) -> bool:
    """Check scalar presence without treating false and zero as missing."""
    return value is not None and np.isscalar(value) and not bool(pd.isna(value))


def _boolean(value: Any) -> bool | None:
    """Parse a schema-declared Boolean while preserving missing/invalid values."""
    if isinstance(value, str):
        return {"true": True, "false": False, "1": True, "0": False}.get(value.lower())
    if _present(value) and value in (False, True):
        return bool(value)
    return None


def _number(value: Any) -> float:
    """Parse finite scalar metadata, ignoring malformed optional JSON fields."""
    if not _present(value) or isinstance(value, (bool, np.bool_)):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _is_boolean(entry: dict) -> bool:
    options = entry.get("options", entry.get("values", ()))
    return bool(options) and all(isinstance(option, bool) for option in options) or isinstance(
        entry.get("default"), bool
    )


def _selector_options(entry: dict) -> tuple[Any, ...]:
    """Retain resolved selector types, including fixed Boolean/integer defaults."""
    options = entry.get("options") or entry.get("values")
    if options:
        return tuple(options)
    return (entry["default"],) if "type" not in entry and "default" in entry else ()


def _recorded_case_key(key: Any, options: tuple[Any, ...]) -> Any:
    """Restore JSON object keys without conflating string and typed selectors."""
    if not isinstance(key, str) or key in (option for option in options if isinstance(option, str)):
        return key
    return next(
        (option for option in options if not isinstance(option, str) and str(option).lower() == key.lower()),
        key,
    )


class _SchemaResolver:
    """Resolve recorded cases and activation ancestry without runtime defaults."""

    def __init__(self, schema: dict) -> None:
        self.schema = compile_association_schema(schema)
        self.entries: dict[str, dict] = {}

        def visit(entries: dict) -> None:
            for name, entry in entries.items():
                if isinstance(entry, dict):
                    self.entries[name] = entry
                    if isinstance(entry.get("activates"), dict):
                        visit(entry["activates"])

        visit(self.schema)
        _, _, self.parents = conditional_yaml_tree(self.schema)

    def resolve(self, row: dict | None) -> dict[str, tuple[dict, bool, str, Any]]:
        """Return resolved entry, activation, parent label and value per parameter."""
        resolved: dict[str, tuple[dict, bool, str, Any]] = {}
        pending: set[str] = set()

        def visit(name: str) -> tuple[dict, bool, str, Any]:
            if name in resolved:
                return resolved[name]
            if name in pending:
                raise ValueError(f"Cyclic recorded tuning dependency involving {name!r}.")
            pending.add(name)
            entry = self.entries[name]
            active, labels = True, []
            if name in self.parents:
                parent = self.parents[name]
                parent_entry, parent_active, parent_branch, parent_value = visit(parent)
                active = parent_active and _boolean(parent_value) is True
                if parent_branch != "all":
                    labels.append(parent_branch)
                labels.append(f"{parent}={parent_value}")
                entry = parent_entry.get("activates", {}).get(name, entry)
            if "depends_on" in entry:
                parent = entry["depends_on"]
                parent_entry, _, parent_branch, parent_value = visit(parent)
                if parent_branch != "all":
                    labels.append(parent_branch)
                labels.append(f"{parent}={parent_value}")
                if not _present(parent_value):
                    active = False
                # JSON object keys stringify Boolean/integer categorical cases.
                options = _selector_options(parent_entry)
                cases = {
                    _recorded_case_key(key, options): value
                    for key, value in entry.get("cases", {}).items()
                }
                entry = resolve_entry({**entry, "cases": cases}, parent_value)
            value = entry.get("default") if row is None else row.get(name)
            if not _present(value) and "type" not in entry:
                value = entry.get("default")
            if _is_boolean(entry):
                value = _boolean(value)
            elif isinstance(value, np.generic):
                value = value.item()
            # iterrows can promote integer selectors to float in numeric-only
            # frames. Recover the declared categorical type for strict cases.
            options = _selector_options(entry)
            if options and not isinstance(value, bool) and np.isfinite(_number(value)):
                value = next(
                    (
                        option for option in options
                        if isinstance(option, (int, float)) and not isinstance(option, bool)
                        and option == _number(value)
                    ),
                    value,
                )
            result = (entry, active, ", ".join(dict.fromkeys(labels)) or "all", value)
            resolved[name] = result
            pending.remove(name)
            return result

        for name in self.entries:
            visit(name)
        return resolved


def analyze_parameters(
    frame: pd.DataFrame, schema: dict, *, notes: list[str] | None = None
) -> dict[str, ParameterSamples]:
    """Collect active searchable values; omit fixed metadata and unusable pairs.

    Observed constants remain available to summaries, while plot callers can
    omit them. Only schema-declared Boolean fields accept Boolean coercion.
    """
    if not schema or "HOTA" not in frame:
        return {}
    resolver = _SchemaResolver(schema)
    observations: dict[str, list[dict]] = {}
    indices: dict[str, list] = {}
    parameters: dict[str, ParameterSamples] = {}
    representations: dict[str, set[tuple[str, str]]] = {}
    for index, row in frame.iterrows():
        score = pd.to_numeric(row["HOTA"], errors="coerce")
        if not np.isfinite(score):
            continue
        for name, (entry, active, branch, value) in resolver.resolve(row.to_dict()).items():
            distribution = entry.get("type")
            if not active or not distribution or not _present(value):
                continue
            categorical = distribution in {"choice", "grid_search"}
            if not categorical:
                value = pd.to_numeric(value, errors="coerce")
                if not np.isfinite(value):
                    continue
            elif isinstance(value, (int, float)) and not np.isfinite(value):
                continue
            kind = "categorical" if categorical else "numeric"
            scale = "log" if not categorical and "log" in distribution else "linear"
            representations.setdefault(name, set()).add((kind, scale))
            parameter = parameters.setdefault(
                name,
                ParameterSamples(name, kind, pd.DataFrame(), scale=scale),
            )
            options = entry.get("options", entry.get("values", ()))
            parameter.options = tuple(dict.fromkeys((*parameter.options, *options)))
            if categorical:
                parameter.branch_options[branch] = tuple(options)
            bounds = entry.get("range")
            if bounds is not None and len(bounds) >= 2:
                low, high = float(bounds[0]), float(bounds[1])
                if np.isfinite([low, high]).all() and low < high:
                    parameter.branch_bounds[branch] = (low, high)
            observations.setdefault(name, []).append(
                {"value": value, "HOTA": score, "trial_id": str(row.get("trial_id", index)), "branch": branch}
            )
            indices.setdefault(name, []).append(index)
    for name, modes in representations.items():
        if len(modes) > 1:
            parameters.pop(name)
            modes_label = ", ".join(f"{kind}/{scale}" for kind, scale in sorted(modes))
            message = f"Parameter {name!r} omitted: observed branches use incompatible distributions ({modes_label})."
            LOGGER.warning(message)
            if notes is not None:
                notes.append(message)
    for name, parameter in parameters.items():
        parameter.samples = pd.DataFrame(observations[name], index=indices[name])
        if parameter.branch_bounds:
            parameter.bounds = (
                min(low for low, _ in parameter.branch_bounds.values()),
                max(high for _, high in parameter.branch_bounds.values()),
            )
    return parameters


def _read_result(path: Path) -> dict:
    """Read the last valid scored result, tolerating interrupted JSONL writes."""
    try:
        content = path.read_text()
    except OSError:
        return {}
    try:
        candidates = [json.loads(content)]
    except json.JSONDecodeError:
        candidates = []
        for line in content.splitlines():
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for result in reversed(candidates):
        if (
            isinstance(result, dict)
            and result.get("done") is not False
            and not result.get("error")
            and np.isfinite(_number(result.get("HOTA")))
        ):
            return result
    return {}


def _metric_view(result: dict) -> dict:
    """Select the recorded class/aggregate, never average scalar HOTA values."""
    from boxmot.engine.eval.results import _select_plot_metrics_data

    validation = result.get("_validation", {})
    if not isinstance(validation, dict):
        return {}
    raw = validation.get("raw", {})
    if not isinstance(raw, dict):
        return {}
    label = validation.get("summary_label")
    if label == "single_class":
        return raw
    if isinstance(label, str) and label:
        view = raw.get(label, {})
        return view if isinstance(view, dict) else {}
    return _select_plot_metrics_data(raw)[1]


def _matches(value: Any, expected: Any) -> bool:
    """Compare recorded defaults without accepting missing values or loose flags."""
    if not _present(value):
        return False
    if isinstance(expected, bool):
        return _boolean(value) is expected
    if isinstance(expected, (int, float)):
        number = pd.to_numeric(value, errors="coerce")
        return bool(np.isfinite(number) and np.isclose(number, expected, rtol=1e-9, atol=1e-12))
    return value == expected


def _find_baseline(frame: pd.DataFrame, schema: dict, fixed: dict) -> int | None:
    """Verify the authored default configuration using only recorded evidence."""
    if not schema:
        return None
    defaults = {
        name: value
        for name, (entry, active, _branch, value) in _SchemaResolver(schema).resolve(None).items()
        if active and "default" in entry
    }
    defaults.update(fixed)
    if not defaults or any(name not in frame for name in defaults):
        return None
    for index, row in frame.iterrows():
        if all(_matches(row[name], value) for name, value in defaults.items()):
            return int(index)
    return None


def load_tune_analysis(tune_dir: Path) -> TrialAnalysis | None:
    """Combine saved CSV, search schema and optional final-trial JSON metadata."""
    tune_dir = Path(tune_dir)
    try:
        trials = pd.read_csv(tune_dir / "results.csv", dtype={"trial_id": str})
    except (OSError, ValueError, pd.errors.ParserError) as error:
        LOGGER.warning("Cannot read tuning results in %s: %s", tune_dir, error)
        return None
    if "HOTA" not in trials:
        LOGGER.warning("results.csv missing HOTA column, skipping tuning analysis.")
        return None
    for metric in _METRICS:
        if metric in trials:
            trials[metric] = pd.to_numeric(trials[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    trials = trials.loc[trials["HOTA"].notna()].reset_index(drop=True)
    if trials.empty:
        LOGGER.warning("No finite HOTA scores in results.csv, skipping tuning analysis.")
        return None
    if "trial_id" not in trials:
        trials["trial_id"] = trials.index.astype(str)
    notes: list[str] = []
    schema, fixed = {}, {}
    try:
        metadata = json.loads((tune_dir / "search-space.json").read_text())
        if not isinstance(metadata, dict) or not isinstance(metadata.get("schema"), dict):
            raise ValueError("expected a recorded schema mapping")
        schema = metadata["schema"]
        fixed = metadata.get("fixed_options", {})
        if not isinstance(fixed, dict):
            fixed = {}
    except (OSError, ValueError) as error:
        notes.append(f"Search schema unavailable; parameter diagnostics skipped ({error}).")

    paths = {path.parent.name.removeprefix("trial_"): path for path in tune_dir.glob("trial_*/result.json")}
    sequences: dict[int, dict[str, float]] = {}
    timestamps: list[float] = []
    missing = 0
    for index, row in trials.iterrows():
        path = paths.get(str(row["trial_id"]))
        result = _read_result(path) if path is not None else {}
        missing += not bool(result)
        timestamp = _number(result.get("timestamp", row.get("timestamp")))
        timestamps.append(timestamp)
        config = result.get("config")
        if isinstance(config, dict):
            for name, value in normalize_trial_config(config).items():
                if name not in trials:
                    trials[name] = pd.Series(index=trials.index, dtype=object)
                if not _present(trials.at[index, name]):
                    trials.at[index, name] = value
        view = _metric_view(result)
        for metric in _METRICS:
            value = _number(view.get(metric))
            if np.isfinite(value) and (metric not in trials or not _present(trials.at[index, metric])):
                trials.at[index, metric] = value
        per_sequence = view.get("per_sequence", {})
        sequences[index] = {
            str(name): float(score)
            for name, metrics in (per_sequence.items() if isinstance(per_sequence, dict) else ())
            if isinstance(metrics, dict) and np.isfinite(score := _number(metrics.get("HOTA")))
        }
    if missing:
        notes.append(f"Final trial metadata unavailable for {missing}/{len(trials)} trials.")
    chronological = bool(np.isfinite(timestamps).all())
    trials["completion_timestamp"] = timestamps
    order = trials["completion_timestamp"].sort_values(kind="stable").index if chronological else trials.index
    sequence_hota = pd.DataFrame.from_dict(sequences, orient="index").reindex(order).reset_index(drop=True)
    trials = trials.loc[order].reset_index(drop=True)
    if not chronological:
        notes.append("Completion timestamps incomplete; trial progression uses recorded CSV order.")
    if "DetA" not in trials or not trials["DetA"].notna().any():
        notes.append("Recorded DetA unavailable; DetA versus AssA diagnostic skipped.")
    elif trials["DetA"].isna().any():
        notes.append("Recorded DetA is incomplete; unavailable metric comparisons are omitted.")
    if sequence_hota.empty:
        notes.append(
            "Per-sequence HOTA unavailable for the selected summary class/aggregate; sequence comparison skipped."
        )
    elif sequence_hota.isna().any(axis=1).any():
        notes.append("Per-sequence HOTA is incomplete; unavailable comparisons are omitted.")
    try:
        parameters = analyze_parameters(trials, schema, notes=notes)
        baseline = _find_baseline(trials, schema, fixed)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        parameters, baseline = {}, None
        notes.append(f"Recorded schema could not be resolved; parameter diagnostics skipped ({error}).")
    label = "Verified default baseline"
    if baseline is None:
        baseline = 0
        label = "First completed trial" if chronological else "First recorded trial"
        notes.append(f"Recorded defaults could not be verified; reference is the {label.lower()}.")
    best = int(trials["HOTA"].idxmax())
    return TrialAnalysis(
        trials=trials,
        parameters=parameters,
        baseline_index=baseline,
        baseline_label=label,
        best_index=best,
        near_best=near_best_mask(trials["HOTA"]),
        sequence_hota=sequence_hota,
        notes=notes,
        chronological=chronological,
        schema=schema,
    )
