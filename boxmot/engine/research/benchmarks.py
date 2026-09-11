from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from boxmot.datasets.config import resolve_dataset_storage_root
from boxmot.engine.config.experiments import resolve_experiment_config
from boxmot.engine.materialization.catalog import STILL_FRAME_EXTENSIONS, resolve_dataset_split_root
from boxmot.engine.tracking.sources import is_appledouble_file


def _resolve_experiment_runtime(
    experiment: str | Path,
    *,
    data_root: str | Path | None = None,
) -> tuple[Path, str, str, str, dict[str, Any]]:
    resolved = resolve_experiment_config(experiment, mode="research")
    dataset = resolved["dataset"]
    dataset_root = resolve_dataset_storage_root(dataset, data_root)
    if dataset.get("layout") == "sequence":
        source_root = resolve_dataset_split_root(dataset, dataset["split"], data_root)
    else:
        relative = PurePosixPath(str(dataset["split_path"]))
        source_root = dataset_root.joinpath(*relative.parts)

    return (
        source_root,
        str(resolved["id"]),
        str(dataset["id"]),
        str(dataset["id"]),
        resolved,
    )


def _discover_sequences(source_root: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for seq_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        image_dir = seq_dir / "img1" if (seq_dir / "img1").is_dir() else seq_dir
        if not any(
            path.is_file() and not is_appledouble_file(path) and path.suffix.lower() in STILL_FRAME_EXTENSIONS
            for path in image_dir.iterdir()
        ):
            continue
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
