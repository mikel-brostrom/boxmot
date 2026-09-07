#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./repotext.sh [context|path] [output-file]

Create a repo-to-text snapshot for a curated ReID ablation context.

Arguments:
  context      Curated context to include: csl_tinyvit_7m,
               csl_tinyvit_11m, csl_tinyvit_23m, or mobilenetv4. Defaults to
               csl_tinyvit_11m. Short aliases include 7m, 11m, 23m,
               and mobilenet.
               The generic csl/tinyvit aliases select 11M.
               mobilenetv4 includes every registered MobileNetV4 variant,
               its shared ReID head/fusion code, recipes, tests, and run data.
  path         File or directory to include. If this is not a known context,
               the script keeps the old behavior and snapshots that path.
  output-file  Output path. Defaults to repo-to-text_<UTC timestamp>.txt.

Environment:
  REPOTEXT_CONTEXT    Curated context when no context argument is passed.
                      Defaults to csl_tinyvit_11m.
  REPOTEXT_MAX_BYTES  Maximum file size to include. Defaults to 1048576.
  REPOTEXT_TOP_ABLATION_RUNS
                      Number of individual ablation runs in the top-runs table.
                      Defaults to 25.
  PYTHON              Python executable. Defaults to python3.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

context="${REPOTEXT_CONTEXT:-csl_tinyvit_11m}"
target="${1:-.}"
timestamp="$(date -u '+%Y-%m-%d-%H-%M-%S-UTC')"
output="${2:-repo-to-text_${timestamp}.txt}"
python_bin="${PYTHON:-python3}"

case "${1:-}" in
  7m|csl_tinyvit_7m|csl-tinyvit-7m)
    context="csl_tinyvit_7m"
    target="."
    output="${2:-repo-to-text_${timestamp}.txt}"
    ;;
  csl|tinyvit|csl_tinyvit|csl-tinyvit|csl_vit_tiny|csl-vit-tiny|11m|csl_tinyvit_11m|csl-tinyvit-11m)
    context="csl_tinyvit_11m"
    target="."
    output="${2:-repo-to-text_${timestamp}.txt}"
    ;;
  23m|csl_tinyvit_23m|csl-tinyvit-23m)
    context="csl_tinyvit_23m"
    target="."
    output="${2:-repo-to-text_${timestamp}.txt}"
    ;;
  mobilenet|mobilenetv4|mobilenet_v4)
    context="mobilenetv4"
    target="."
    output="${2:-repo-to-text_${timestamp}.txt}"
    ;;
esac

"${python_bin}" - "${target}" "${output}" "${context}" <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


target = Path(sys.argv[1]).expanduser().resolve()
output = Path(sys.argv[2]).expanduser().resolve()
requested_context = sys.argv[3]
max_bytes = int(os.environ.get("REPOTEXT_MAX_BYTES", "1048576"))
top_ablation_runs_limit = int(os.environ.get("REPOTEXT_TOP_ABLATION_RUNS", "25"))
COMMON_SOURCE_PATHS = (
    "boxmot/reid/training",
    "boxmot/engine/configs/runtime.yaml",
)
CSL_LOG_PATTERNS = (
    "runs/ablation_csl_tinyvit*/**/*.log",
)
CSL_7M_RUN_PATTERNS = tuple(
    f"runs/{run_prefix}/**/{artifact}"
    for run_prefix in ("csl_tinyvit_7m*", "ablation_csl_tinyvit_7m*")
    for artifact in ("metrics.json", "hparams.json", "*.log")
)
CSL_11M_RUN_DIRS = (
    "csl_tinyvit_11m_anatomical_grid",
    "csl_tinyvit_11m_market1501_gray",
    "csl_tinyvit_11m_pav",
)
CSL_11M_RUN_ARTIFACT_PATTERNS = tuple(
    f"runs/{run_dir}/**/{artifact}"
    for run_dir in CSL_11M_RUN_DIRS
    for artifact in ("metrics.json", "hparams.json", "*.log")
)
CSL_11M_METRICS_PATTERNS = tuple(
    f"runs/{run_dir}/**/metrics.json"
    for run_dir in CSL_11M_RUN_DIRS
)
MOBILENETV4_RUN_ARTIFACT_PATTERNS = tuple(
    f"runs/{run_prefix}/**/{artifact}"
    for run_prefix in ("ablation_mobilenetv4*", "reid_train")
    for artifact in ("metrics.json", "hparams.json", "*.log")
)
CONTEXTS = {
    "csl_tinyvit_7m": {
        "label": "csl-tinyvit-7m-ablation-context",
        "models": (
            "csl_tinyvit_7m",
            "csl_tinyvit_7m_v20",
        ),
        "source_paths": COMMON_SOURCE_PATHS
        + (
            "boxmot/reid/training/configs/recipes/csl_tinyvit_7m_v20.yaml",
            "boxmot/reid/backbones/families/csl_tinyvit",
        ),
        "patterns": (
            ("ablation_csl_tinyvit_7m*.sh",)
            + CSL_LOG_PATTERNS
            + CSL_7M_RUN_PATTERNS
        ),
        "ablation_dir_patterns": (
            "csl_tinyvit_7m*",
            "ablation_csl_tinyvit_7m*",
        ),
        "metrics_patterns": (
            "runs/csl_tinyvit_7m*/**/metrics.json",
            "runs/ablation_csl_tinyvit_7m*/**/metrics.json",
        ),
    },
    "csl_tinyvit_11m": {
        "label": "csl-tinyvit-11m-ablation-context",
        "model": "csl_tinyvit_11m",
        "source_paths": COMMON_SOURCE_PATHS
        + (
            "boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml",
            "boxmot/reid/backbones/families/csl_tinyvit",
        ),
        "patterns": (
            ("ablation_csl_tinyvit_11m*.sh",)
            + CSL_LOG_PATTERNS
            + CSL_11M_RUN_ARTIFACT_PATTERNS
        ),
        "ablation_dir_patterns": (
            "ablation_csl_tinyvit_11m*",
            "ablation_csl_tinyvit*",
            *CSL_11M_RUN_DIRS,
        ),
        "metrics_patterns": (
            "runs/ablation_csl_tinyvit*/**/metrics.json",
            *CSL_11M_METRICS_PATTERNS,
        ),
    },
    "csl_tinyvit_23m": {
        "label": "csl-tinyvit-23m-ablation-context",
        "model": "csl_tinyvit_23m",
        "source_paths": COMMON_SOURCE_PATHS
        + (
            "boxmot/reid/training/configs/recipes/csl_tinyvit_23m.yaml",
            "boxmot/reid/backbones/families/csl_tinyvit",
        ),
        "patterns": (
            "ablation_csl_tinyvit_23m*.sh",
            "ablation_csl_tinyvit_metric_alignment.sh",
            "ablation_csl_tinyvit_reid_losses.sh",
        )
        + CSL_LOG_PATTERNS,
        "ablation_dir_patterns": ("ablation_csl_tinyvit_23m*", "ablation_csl_tinyvit*"),
        "metrics_patterns": ("runs/ablation_csl_tinyvit*/**/metrics.json",),
    },
    "mobilenetv4": {
        "label": "mobilenetv4-ablation-context",
        "model_prefix": "mobilenetv4_",
        "source_paths": COMMON_SOURCE_PATHS
        + (
            "boxmot/engine/config.py",
            "boxmot/engine/cli.py",
            "boxmot/reid/backbones/base.py",
            "boxmot/reid/backbones/head_registry.py",
            "boxmot/reid/backbones/option_registry.py",
            "boxmot/reid/backbones/mobilenetv4.py",
            "boxmot/reid/backbones/registry.py",
            "boxmot/reid/backbones/families/csl_tinyvit/blocks.py",
            "boxmot/reid/backbones/families/csl_tinyvit/fusion.py",
            "boxmot/reid/backbones/families/csl_tinyvit/heads.py",
            "boxmot/reid/backbones/families/csl_tinyvit/pooling.py",
            "boxmot/reid/core/registry.py",
            "docs/modes/train.md",
            "tests/performance/benchmark_reid_inference.py",
            "tests/unit/reid/backbones/test_mobilenetv4_medium_v20.py",
        ),
        "patterns": (
            "boxmot/reid/training/configs/recipes/mobilenetv4*.yaml",
            "ablation_mobilenetv4*.sh",
            "compare_reid_inference_speed.sh",
        )
        + MOBILENETV4_RUN_ARTIFACT_PATTERNS,
        "ablation_dir_patterns": (
            "ablation_mobilenetv4*",
            "reid_train/**/mobilenetv4*",
        ),
        "metrics_patterns": (
            "runs/ablation_mobilenetv4*/**/metrics.json",
            "runs/reid_train/**/metrics.json",
        ),
    },
}
CONTEXT_ALIASES = {
    "7m": "csl_tinyvit_7m",
    "csl-tinyvit-7m": "csl_tinyvit_7m",
    "csl": "csl_tinyvit_11m",
    "tinyvit": "csl_tinyvit_11m",
    "csl_tinyvit": "csl_tinyvit_11m",
    "csl-tinyvit": "csl_tinyvit_11m",
    "csl_vit_tiny": "csl_tinyvit_11m",
    "csl-vit-tiny": "csl_tinyvit_11m",
    "11m": "csl_tinyvit_11m",
    "csl-tinyvit-11m": "csl_tinyvit_11m",
    "23m": "csl_tinyvit_23m",
    "csl-tinyvit-23m": "csl_tinyvit_23m",
    "mobilenet": "mobilenetv4",
    "mobilenet_v4": "mobilenetv4",
}
context_name = CONTEXT_ALIASES.get(requested_context, requested_context)
if context_name not in CONTEXTS:
    choices = ", ".join(sorted(CONTEXTS))
    raise SystemExit(f"Unknown REPOTEXT_CONTEXT={requested_context!r}; expected one of: {choices}")
context = CONTEXTS[context_name]
skip_dirs = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site",
}
skip_suffixes = {
    ".avif",
    ".bin",
    ".db",
    ".engine",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".npy",
    ".pdf",
    ".png",
    ".pyc",
    ".pt",
    ".pth",
    ".sqlite",
    ".torchscript",
    ".webp",
    ".xlsx",
    ".xls",
    ".zip",
}


def git_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path if path.is_dir() else path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return Path(result.stdout.strip()).resolve()


def is_skipped(path: Path) -> bool:
    return (
        any(part in skip_dirs for part in path.parts)
        or path.suffix.lower() in skip_suffixes
        or path.name.startswith("repo-to-text_")
    )


def tracked_or_unignored_files(path: Path) -> list[Path]:
    root = git_root(path)
    if root is None:
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        return [p.resolve() for p in candidates if not is_skipped(p)]

    rel = path.relative_to(root) if path != root else Path(".")
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "--", str(rel)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    files: list[Path] = []
    for line in result.stdout.splitlines():
        candidate = (root / line).resolve()
        if not candidate.is_file() or is_skipped(candidate):
            continue
        try:
            candidate.relative_to(path if path.is_dir() else path.parent)
        except ValueError:
            continue
        files.append(candidate)
    return sorted(files)


def hparams_model(path: Path) -> str | None:
    """Read the canonical model name recorded beside an experiment artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run = payload.get("run")
    if isinstance(run, dict) and run.get("model_name"):
        return str(run["model_name"])
    model_name = payload.get("model_name")
    if model_name:
        return str(model_name)
    model = payload.get("model")
    return str(model) if isinstance(model, str) else None


def experiment_model(path: Path, root: Path) -> str | None:
    """Resolve the model for a run artifact from its nearest hparams file."""
    runs_root = root / "runs"
    current = path.parent
    while current != runs_root and runs_root in current.parents:
        hparams = current / "hparams.json"
        if hparams.is_file():
            return hparams_model(hparams)
        current = current.parent
    return None


def context_accepts_model(model_name: str | None) -> bool:
    """Return whether a run model belongs to the selected curated context."""
    selected_model = context.get("model")
    selected_models = context.get("models", ())
    selected_prefix = context.get("model_prefix")
    if not selected_model and not selected_models and not selected_prefix:
        return True
    if not model_name:
        return False
    return bool(
        (selected_model and model_name == selected_model)
        or (selected_models and model_name in selected_models)
        or (selected_prefix and model_name.startswith(str(selected_prefix)))
    )


def matches_selected_model(path: Path, root: Path) -> bool:
    if not context.get("model") and not context.get("models") and not context.get("model_prefix"):
        return True
    try:
        path.relative_to(root / "runs")
    except ValueError:
        return True
    return context_accepts_model(experiment_model(path, root))


def curated_files(root: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        candidate = candidate.resolve()
        if (
            candidate in seen
            or not candidate.is_file()
            or is_skipped(candidate)
            or not matches_selected_model(candidate, root)
        ):
            return
        seen.add(candidate)
        files.append(candidate)

    for rel_path in context["source_paths"]:
        source_path = root / rel_path
        if source_path.is_dir():
            for candidate in tracked_or_unignored_files(source_path):
                add(candidate)
        else:
            add(source_path)
    for pattern in context["patterns"]:
        for candidate in root.glob(pattern):
            add(candidate)
    return sorted(files)


def as_float(value: object, default: float = float("-inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_metric(value: object) -> str:
    number = as_float(value)
    if number == float("-inf"):
        return "-"
    return f"{number:.4f}"


def ablation_sort_key(row: dict[str, object]) -> tuple[float, float, float, str, str]:
    return (
        as_float(row["best_mAP"]),
        as_float(row["best_rank1"]),
        as_float(row["best_epoch"]),
        str(row["ablation"]),
        str(row["run"]),
    )


def ablation_run_rows(root: Path) -> list[dict[str, object]]:
    runs_root = root / "runs"
    rows: list[dict[str, object]] = []

    metrics_paths = sorted(
        {
            metrics_path
            for pattern in context["metrics_patterns"]
            for metrics_path in root.glob(pattern)
        }
    )
    for metrics_path in metrics_paths:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recorded_model = experiment_model(metrics_path, root)
        if not context_accepts_model(recorded_model):
            continue

        rel = metrics_path.relative_to(root)
        ablation = rel.parts[1]
        run_rel = metrics_path.parent.relative_to(runs_root / ablation)
        rows.append(
            {
                "ablation": ablation,
                "run": str(run_rel),
                "run_path": str(metrics_path.parent.relative_to(root)),
                "model": recorded_model or metrics.get("model", "-"),
                "dataset": metrics.get("dataset", "-"),
                "best_epoch": metrics.get("best_epoch", "-"),
                "best_mAP": metrics.get("best_mAP"),
                "best_rank1": metrics.get("best_rank1"),
            }
        )

    return rows


def ablation_top_rows(root: Path) -> list[dict[str, object]]:
    rows = sorted(ablation_run_rows(root), key=ablation_sort_key, reverse=True)
    if top_ablation_runs_limit <= 0:
        return rows
    return rows[:top_ablation_runs_limit]


def ablation_best_rows(root: Path) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in ablation_run_rows(root):
        groups.setdefault(str(row["ablation"]), []).append(row)

    rows: list[dict[str, object]] = []
    for ablation, candidates in groups.items():
        best = dict(max(candidates, key=ablation_sort_key))
        best["metrics_count"] = len(candidates)
        rows.append(best)

    return sorted(rows, key=ablation_sort_key, reverse=True)


def ablation_best_table(root: Path) -> str:
    run_rows = ablation_run_rows(root)
    if not run_rows:
        return "No ablation metrics found.\n"

    lines = [
        "Top Ablation Runs:",
        "| Rank | Ablation | Run | mAP | Rank-1 | Epoch | Model | Dataset |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(ablation_top_rows(root), start=1):
        lines.append(
            "| {rank} | {ablation} | {run} | {mAP} | {rank1} | {epoch} | {model} | {dataset} |".format(
                rank=rank,
                ablation=row["ablation"],
                run=row["run"],
                mAP=fmt_metric(row["best_mAP"]),
                rank1=fmt_metric(row["best_rank1"]),
                epoch=row["best_epoch"],
                model=row["model"],
                dataset=row["dataset"],
            )
        )
    limit = "all" if top_ablation_runs_limit <= 0 else str(min(top_ablation_runs_limit, len(run_rows)))
    lines.append("")
    lines.append(
        f"Top Ablation Runs shows {limit} of {len(run_rows)} {context_name} metrics files, selected by best_mAP, then best_rank1, then best_epoch."
    )
    lines.append("")
    lines.append("Best Run Per Ablation:")
    lines.extend(
        [
        "| Ablation | Best run | mAP | Rank-1 | Epoch | Model | Dataset | Metrics files |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for row in ablation_best_rows(root):
        lines.append(
            "| {ablation} | {run} | {mAP} | {rank1} | {epoch} | {model} | {dataset} | {count} |".format(
                ablation=row["ablation"],
                run=row["run"],
                mAP=fmt_metric(row["best_mAP"]),
                rank1=fmt_metric(row["best_rank1"]),
                epoch=row["best_epoch"],
                model=row["model"],
                dataset=row["dataset"],
                count=row["metrics_count"],
            )
        )
    lines.append("")
    patterns = ", ".join(f"runs/{pattern}" for pattern in context["ablation_dir_patterns"])
    lines.append(f"Best Run Per Ablation keeps only one winner per {patterns} directory.")
    return "\n".join(lines) + "\n"


def read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if len(data) > max_bytes or b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def tree_lines(files: list[Path], base: Path) -> list[str]:
    if not files:
        return ["."]
    rels = [p.relative_to(base if base.is_dir() else base.parent) for p in files]
    lines = ["."]
    directories = sorted({parent for rel in rels for parent in rel.parents if str(parent) != "."})
    for directory in directories:
        lines.append(str(directory))
    for rel in rels:
        lines.append(str(rel))
    return lines


root = git_root(target) or git_root(Path.cwd()) or Path.cwd().resolve()
use_curated_scope = target == root or target == Path.cwd().resolve()
if not use_curated_scope and not target.exists():
    raise SystemExit(
        f"Snapshot path does not exist: {target}. "
        "Use --help to list curated context aliases."
    )
base = root if use_curated_scope else target if target.is_dir() else target.parent
files = curated_files(root) if use_curated_scope else tracked_or_unignored_files(target)
included: list[tuple[Path, str]] = []
for file_path in files:
    content = read_text(file_path)
    if content is not None:
        included.append((file_path, content))

label = str(context["label"]) if use_curated_scope else target.name if target.name else str(target)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write("<repo-to-text>\n")
    handle.write(f"Directory: {label}\n\n")
    if use_curated_scope:
        handle.write("Ablation Best Results:\n")
        handle.write("<ablation_best_results>\n")
        handle.write(ablation_best_table(root))
        handle.write("</ablation_best_results>\n\n")
    handle.write("Directory Structure:\n")
    handle.write("<directory_structure>\n")
    for line in tree_lines([path for path, _ in included], base):
        handle.write(f"{line}\n")
    handle.write("\n</directory_structure>\n\n")
    for file_path, content in included:
        rel = file_path.relative_to(base)
        handle.write(f'<content full_path="{rel}">\n')
        handle.write(content.rstrip())
        handle.write("\n\n</content>\n\n")
    handle.write("</repo-to-text>\n")

print(f"Wrote {output}")
PY
