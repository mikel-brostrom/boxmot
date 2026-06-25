from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from boxmot.configs import DEFAULT_DETECTOR, DEFAULT_REID
from boxmot.configs.benchmark import (
    apply_benchmark_config,
    resolve_required_reid_model,
    resolve_required_yolo_model,
)
from boxmot.data.dataset import _collect_seq_info
from boxmot.utils.misc import resolve_model_path


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


