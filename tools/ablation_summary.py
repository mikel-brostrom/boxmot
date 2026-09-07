from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _percent(value: Any) -> float | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return parsed * 100.0 if parsed <= 1.0 else parsed


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_load_error": str(exc)}


def _best_from_val_rows(rows: list[dict[str, Any]]) -> tuple[int | None, float | None, float | None]:
    best_epoch: int | None = None
    best_map: float | None = None
    best_rank1: float | None = None
    for row in rows:
        epoch = _as_int(row.get("epoch"))
        metrics_blocks = [value for key, value in row.items() if key != "epoch" and isinstance(value, dict)]
        if not metrics_blocks:
            metrics_blocks = [row]
        for metrics in metrics_blocks:
            map_value = _percent(metrics.get("mAP"))
            rank1_value = _percent(metrics.get("rank1", metrics.get("R1")))
            if map_value is None:
                continue
            if best_map is None or map_value > best_map:
                best_epoch = epoch
                best_map = map_value
                best_rank1 = rank1_value
    return best_epoch, best_map, best_rank1


def _hparams_summary(exp_dir: Path) -> dict[str, Any]:
    hparams_path = exp_dir / "hparams.json"
    if not hparams_path.is_file():
        return {}
    hparams = _load_json(hparams_path)
    if "_load_error" in hparams:
        return {"hparams_error": hparams["_load_error"]}
    model = hparams.get("model") if isinstance(hparams.get("model"), dict) else {}
    data = hparams.get("data") if isinstance(hparams.get("data"), dict) else {}
    return {
        "dataset": data.get("dataset"),
        "model": (hparams.get("run") or {}).get("model_name") if isinstance(hparams.get("run"), dict) else None,
        "feature_fusion": model.get("feature_fusion"),
        "post_fusion_mixer": (model.get("post_fusion_mixer") or {}).get("mode")
        if isinstance(model.get("post_fusion_mixer"), dict)
        else None,
    }


def _experiment_summary(exp_dir: Path) -> dict[str, Any]:
    metrics_path = exp_dir / "metrics.json"
    result: dict[str, Any] = {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "metrics_path": str(metrics_path),
        "valid": False,
        "issues": [],
        "epochs": None,
        "last_epoch": 0,
        "best_epoch": None,
        "best_mAP": None,
        "best_rank1": None,
    }
    result.update(_hparams_summary(exp_dir))

    if not metrics_path.is_file():
        result["issues"].append("missing metrics.json")
        return result

    metrics = _load_json(metrics_path)
    if "_load_error" in metrics:
        result["issues"].append(f"invalid metrics.json: {metrics['_load_error']}")
        return result

    train_rows = metrics.get("train") if isinstance(metrics.get("train"), list) else []
    val_rows = metrics.get("val") if isinstance(metrics.get("val"), list) else []
    last_epoch = _as_int(train_rows[-1].get("epoch")) if train_rows else None
    target_epochs = _as_int(metrics.get("epochs"))
    best_epoch = _as_int(metrics.get("best_epoch"))
    best_map = _percent(metrics.get("best_mAP"))
    best_rank1 = _percent(metrics.get("best_rank1"))

    if best_map is None:
        val_best_epoch, best_map, best_rank1 = _best_from_val_rows(val_rows)
        if best_epoch is None:
            best_epoch = val_best_epoch

    result.update(
        {
            "epochs": target_epochs,
            "last_epoch": last_epoch or 0,
            "best_epoch": best_epoch,
            "best_mAP": best_map,
            "best_rank1": best_rank1,
        }
    )

    if target_epochs and last_epoch and last_epoch < target_epochs:
        result["issues"].append(f"incomplete {last_epoch}/{target_epochs}")
    if not train_rows:
        result["issues"].append("missing train metrics")
    if best_map is None:
        result["issues"].append("missing best mAP")

    result["valid"] = not result["issues"]
    return result


def load_experiments(project: str | Path) -> list[dict[str, Any]]:
    project_path = Path(project)
    if not project_path.exists():
        return []
    experiments = [
        _experiment_summary(path)
        for path in sorted(project_path.iterdir())
        if path.is_dir()
    ]
    return experiments


def _fmt_percent(value: Any) -> str:
    parsed = _as_float(value)
    return "-" if parsed is None else f"{parsed:.2f}%"


def _fmt_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _status(result: dict[str, Any]) -> str:
    if result["valid"]:
        return "complete"
    if result["issues"]:
        return "; ".join(str(issue) for issue in result["issues"])
    return "invalid"


def _render_table(experiments: list[dict[str, Any]]) -> str:
    columns = (
        ("Run", "name"),
        ("Status", "_status"),
        ("Epoch", "_epoch"),
        ("Best", "best_epoch"),
        ("mAP", "best_mAP"),
        ("R1", "best_rank1"),
        ("Fusion", "feature_fusion"),
        ("Mixer", "post_fusion_mixer"),
    )
    rows = []
    for result in experiments:
        target = result.get("epochs")
        last = result.get("last_epoch")
        rows.append(
            {
                "name": result["name"],
                "_status": _status(result),
                "_epoch": f"{last}/{target}" if target else _fmt_value(last),
                "best_epoch": _fmt_value(result.get("best_epoch")),
                "best_mAP": _fmt_percent(result.get("best_mAP")),
                "best_rank1": _fmt_percent(result.get("best_rank1")),
                "feature_fusion": _fmt_value(result.get("feature_fusion")),
                "post_fusion_mixer": _fmt_value(result.get("post_fusion_mixer")),
            }
        )

    widths = {
        title: max(len(title), *(len(str(row[key])) for row in rows))
        for title, key in columns
    }
    lines = []
    header = "  ".join(title.ljust(widths[title]) for title, _ in columns)
    rule = "  ".join("-" * widths[title] for title, _ in columns)
    lines.extend([header, rule])
    for row in rows:
        lines.append("  ".join(str(row[key]).ljust(widths[title]) for title, key in columns))
    return "\n".join(lines)


def _write_artifacts(project: Path, experiments: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = project / "ablation_summary.json"
    md_path = project / "ablation_summary.md"
    json_path.write_text(json.dumps(experiments, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Ablation Summary\n\n"
        f"`{project}`\n\n"
        "```text\n"
        f"{_render_table(experiments)}\n"
        "```\n",
        encoding="utf-8",
    )
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BoxMOT ReID ablation runs.")
    parser.add_argument("project", type=Path, help="Project directory containing experiment subdirectories.")
    args = parser.parse_args()

    experiments = load_experiments(args.project)
    if not experiments:
        raise SystemExit(f"No experiment directories found under {args.project}")

    print(f"Ablation summary: {args.project}")
    print(_render_table(experiments))

    valid = [result for result in experiments if result["valid"] and result.get("best_mAP") is not None]
    if valid:
        best = max(valid, key=lambda result: float(result["best_mAP"]))
        print(
            "\nBest by mAP: "
            f"{best['name']} ({_fmt_percent(best['best_mAP'])}, R1={_fmt_percent(best.get('best_rank1'))})"
        )

    json_path, md_path = _write_artifacts(args.project, experiments)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
