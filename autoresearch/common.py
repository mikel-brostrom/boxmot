from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from boxmot.configs import build_mode_namespace
from boxmot.model import TrackEvalMetrics, TuneResults

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = (REPO_ROOT / "runs").resolve()
DEFAULT_RESULTS_PATH = (Path(__file__).resolve().parent / "results.tsv").resolve()
ARTIFACTS_DIRNAME = "autoresearch"

SUMMARY_FIELDS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
RESULT_FIELDS = (
    "commit",
    "tracker",
    "benchmark",
    "phase",
    "HOTA",
    "MOTA",
    "IDF1",
    "AssA",
    "AssRe",
    "IDSW",
    "IDs",
    "IDSW_rate",
    "status",
    "description",
)
RESULT_STATUSES = ("keep", "discard", "crash")
LEGACY_RESULT_FIELDS = (
    "commit",
    "tracker",
    "benchmark",
    "phase",
    "HOTA",
    "MOTA",
    "IDF1",
    "AssA",
    "AssRe",
    "IDSW",
    "IDs",
    "IDSW_rate",
    "status",
    "description",
    "artifact",
)


def tracker_config_path(tracker: str) -> Path:
    """Return the tracker YAML config path for a tracker name."""
    return REPO_ROOT / "boxmot" / "configs" / "trackers" / f"{str(tracker).lower()}.yaml"


def ensure_tracker_exists(tracker: str) -> str:
    """Validate that the requested tracker has a registered runtime YAML."""
    normalized = str(tracker).lower()
    cfg_path = tracker_config_path(normalized)
    if not cfg_path.exists():
        raise ValueError(f"Unknown tracker {tracker!r}. Expected config at {cfg_path}")
    return normalized


def _clean_override(overrides: dict[str, Any], key: str, value: Any) -> None:
    if value in (None, ""):
        return
    overrides[key] = value


def build_experiment_args(
    mode: str,
    *,
    benchmark: str,
    tracker: str,
    detector: str | Path | None = None,
    reid: str | Path | None = None,
    project: str | Path | None = None,
    name: str = "",
    device: str | None = None,
    imgsz: int | tuple[int, int] | None = None,
    batch_size: int | None = None,
    n_threads: int | None = None,
    classes: list[int] | str | None = None,
    tracking_backend: str = "thread",
    half: bool | None = None,
    conf: float | None = None,
    iou: float | None = None,
    n_trials: int | None = None,
    objectives: list[str] | None = None,
    maximize: list[str] | None = None,
    minimize: list[str] | None = None,
) -> Any:
    """Build a BoxMOT runtime namespace for eval/tune experiments."""
    tracker_name = ensure_tracker_exists(tracker)
    project_path = Path(project).resolve() if project is not None else DEFAULT_PROJECT

    overrides: dict[str, Any] = {
        "tracker": tracker_name,
        "data": benchmark,
        "project": project_path,
        "name": name,
        "tracking_backend": tracking_backend,
    }
    _clean_override(overrides, "detector", detector)
    _clean_override(overrides, "reid", reid)
    _clean_override(overrides, "device", device)
    _clean_override(overrides, "imgsz", imgsz)
    _clean_override(overrides, "batch_size", batch_size)
    _clean_override(overrides, "n_threads", n_threads)
    _clean_override(overrides, "classes", classes)
    _clean_override(overrides, "half", half)
    _clean_override(overrides, "conf", conf)
    _clean_override(overrides, "iou", iou)
    _clean_override(overrides, "n_trials", n_trials)
    _clean_override(overrides, "objectives", objectives)
    _clean_override(overrides, "maximize", maximize)
    _clean_override(overrides, "minimize", minimize)

    return build_mode_namespace(mode, overrides, explicit_keys=set(overrides))


def experiment_dir(project: str | Path, tracker: str, benchmark: str) -> Path:
    """Return the directory where autoresearch artifacts for one tracker live."""
    project_path = Path(project).resolve()
    benchmark_slug = str(benchmark).replace("/", "_")
    return project_path / ARTIFACTS_DIRNAME / str(tracker).lower() / benchmark_slug


def summarize_eval_results(results: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten TrackEval output to one comparable summary row."""
    metrics = TrackEvalMetrics(results or {})
    summary = dict(metrics.summary)
    ids = int(summary.get("IDs", 0) or 0)
    idsw = int(summary.get("IDSW", 0) or 0)

    payload: dict[str, Any] = {
        "summary_name": metrics.summary_name,
        "IDSW_rate": float(summary.get("IDSW_rate", idsw / max(ids, 1))),
    }
    for field in SUMMARY_FIELDS:
        value = summary.get(field, 0)
        if field in {"IDSW", "IDs"}:
            payload[field] = int(value)
        else:
            payload[field] = float(value)
    return payload


def summarize_tune_results(results: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten BoxMOT tune output around the best trial."""
    tune_results = TuneResults(results or {})
    best_metrics = summarize_eval_results(tune_results.best_metrics.to_dict())

    return {
        "tracking_method": tune_results.tracking_method,
        "benchmark": tune_results.benchmark,
        "objectives": tune_results.objectives,
        "maximize": tune_results.maximize,
        "minimize": tune_results.minimize,
        "best_trial_id": tune_results.best_trial_id,
        "best_config": tune_results.best_config,
        "best_metrics": best_metrics,
        "tune_dir": tune_results.tune_dir,
        "results_csv": tune_results.results_csv,
        "summary_path": tune_results.summary_path,
        "best_yaml": tune_results.best_yaml,
    }


def to_jsonable(value: Any) -> Any:
    """Recursively convert Paths and other runtime objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one JSON artifact and return its resolved path."""
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return output_path


def _results_header(fields: tuple[str, ...] = RESULT_FIELDS) -> str:
    return "\t".join(fields)


def _legacy_backup_path(results_path: Path) -> Path:
    backup = results_path.with_name(f"{results_path.stem}.legacy{results_path.suffix}")
    index = 2
    while backup.exists():
        backup = results_path.with_name(f"{results_path.stem}.legacy.{index}{results_path.suffix}")
        index += 1
    return backup


def ensure_results_tsv(path: str | Path = DEFAULT_RESULTS_PATH) -> Path:
    """Create the experiment results TSV if it does not exist yet."""
    results_path = Path(path).resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    expected_header = _results_header()
    if results_path.exists():
        if results_path.stat().st_size == 0:
            results_path.write_text(expected_header + "\n")
            return results_path

        with results_path.open() as handle:
            header = handle.readline().strip()
        if header == expected_header:
            return results_path

        backup_path = _legacy_backup_path(results_path)
        results_path.replace(backup_path)

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
    return results_path


def current_commit_short() -> str:
    """Return the current short git commit hash, or an empty string outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=7", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def sanitize_description(description: str) -> str:
    """Collapse tabs/newlines so every result stays on one TSV row."""
    return " ".join(str(description).replace("\t", " ").split())


def normalize_status(status: str) -> str:
    """Validate one experiment ledger status value."""
    normalized = str(status).strip().lower()
    if normalized not in RESULT_STATUSES:
        choices = ", ".join(RESULT_STATUSES)
        raise ValueError(f"status must be one of: {choices}")
    return normalized


def summarize_logged_metrics(summary: Mapping[str, Any] | None, *, status: str) -> dict[str, Any]:
    """Reduce an eval/tune summary to one full ledger metric payload."""
    normalized_status = normalize_status(status)
    if normalized_status == "crash" or summary is None:
        return {
            "HOTA": 0.0,
            "MOTA": 0.0,
            "IDF1": 0.0,
            "AssA": 0.0,
            "AssRe": 0.0,
            "IDSW": 0,
            "IDs": 0,
            "IDSW_rate": 0.0,
        }

    metrics = {
        "HOTA": float(summary.get("HOTA", 0.0) or 0.0),
        "MOTA": float(summary.get("MOTA", 0.0) or 0.0),
        "IDF1": float(summary.get("IDF1", 0.0) or 0.0),
        "AssA": float(summary.get("AssA", 0.0) or 0.0),
        "AssRe": float(summary.get("AssRe", 0.0) or 0.0),
        "IDSW": int(summary.get("IDSW", 0) or 0),
        "IDs": int(summary.get("IDs", 0) or 0),
    }
    metrics["IDSW_rate"] = float(
        summary.get("IDSW_rate", metrics["IDSW"] / max(metrics["IDs"], 1)) or 0.0
    )
    return metrics


def load_logged_metrics_from_artifact(path: str | Path) -> dict[str, Any]:
    """Load full ledger metadata and metrics from an eval/tune JSON artifact."""
    artifact_path = Path(path).resolve()
    payload = json.loads(artifact_path.read_text())
    mode = str(payload.get("mode", "")).lower()
    if mode == "tune":
        summary = (payload.get("summary") or {}).get("best_metrics") or {}
    else:
        summary = payload.get("summary") or {}
    if not summary:
        raise ValueError(f"No summary metrics found in artifact: {artifact_path}")
    return {
        "tracker": str(payload.get("tracker", "") or ""),
        "benchmark": str(payload.get("benchmark", "") or ""),
        "phase": mode,
        "summary": summarize_logged_metrics(summary, status="keep"),
    }


def append_results_row(
    path: str | Path,
    *,
    summary: Mapping[str, Any] | None,
    status: str,
    tracker: str = "",
    benchmark: str = "",
    phase: str = "",
    description: str = "",
) -> Path:
    """Append one experiment row to the shared autoresearch TSV."""
    results_path = ensure_results_tsv(path)
    normalized_status = normalize_status(status)
    logged_metrics = summarize_logged_metrics(summary, status=normalized_status)
    row = {
        "commit": current_commit_short(),
        "tracker": str(tracker or ""),
        "benchmark": str(benchmark or ""),
        "phase": str(phase or ""),
        "HOTA": f"{logged_metrics['HOTA']:.6f}",
        "MOTA": f"{logged_metrics['MOTA']:.6f}",
        "IDF1": f"{logged_metrics['IDF1']:.6f}",
        "AssA": f"{logged_metrics['AssA']:.6f}",
        "AssRe": f"{logged_metrics['AssRe']:.6f}",
        "IDSW": str(logged_metrics["IDSW"]),
        "IDs": str(logged_metrics["IDs"]),
        "IDSW_rate": f"{logged_metrics['IDSW_rate']:.6f}",
        "status": normalized_status,
        "description": sanitize_description(description),
    }
    with results_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow(row)
    return results_path
