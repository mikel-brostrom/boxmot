"""Command adapter for offline HP-GRD privileged caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parse_extra_json(value: str | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"--extra-json is invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError("--extra-json must decode to a JSON object")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser(
        "index",
        help="Export the exact stable training mapping from a registered dataset",
    )
    index.add_argument("--dataset", required=True, help="Registered ReID dataset name")
    index.add_argument("--data-dir", type=Path, required=True, help="Dataset root used by training")
    index.add_argument("--output", type=Path, required=True, help="Destination JSON index")
    index.add_argument("--overwrite", action="store_true", help="Atomically replace an existing output file")

    build = commands.add_parser("build", help="Build and self-validate a cache from precomputed tensors")
    build.add_argument("--tensor-input", type=Path, required=True, help="torch.save tensor dictionary")
    build.add_argument("--dataset-index", type=Path, required=True, help="JSON/JSONL stable sample index")
    build.add_argument("--teacher-provenance", type=Path, required=True, help="Teacher artifact/config to hash")
    build.add_argument(
        "--part-names",
        nargs="+",
        help="Ordered semantic part names (checked against extractor metadata when present)",
    )
    build.add_argument("--output", type=Path, required=True, help="Destination .pt cache")
    build.add_argument("--extra-json", help="Optional JSON object stored under manifest extra.user")
    build.add_argument("--overwrite", action="store_true", help="Atomically replace an existing output file")

    validate = commands.add_parser("validate", help="Validate cache payload and external provenance")
    validate.add_argument("--cache", type=Path, required=True, help="Cache produced by the build command")
    validate.add_argument("--dataset-index", type=Path, required=True, help="JSON/JSONL stable sample index")
    validate.add_argument("--teacher-provenance", type=Path, required=True, help="Teacher artifact/config to hash")
    validate.add_argument("--manifest-sha256", help="Optional pinned manifest digest")
    validate.add_argument("--part-names", nargs="+", help="Require this exact ordered semantic part axis")
    validate.add_argument(
        "--require-exact-index-file",
        action="store_true",
        help="Also require byte-for-byte identity of the original index file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone cache builder/validator command."""

    args = _build_parser().parse_args(argv)

    from boxmot.reid.training.privileged_cache import (
        build_privileged_cache,
        export_dataset_index,
        validate_privileged_cache,
    )

    if args.command == "index":
        result = export_dataset_index(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            output=args.output,
            overwrite=args.overwrite,
        )
    elif args.command == "build":
        result = build_privileged_cache(
            tensor_input=args.tensor_input,
            dataset_index=args.dataset_index,
            teacher_provenance=args.teacher_provenance,
            output=args.output,
            part_names=args.part_names,
            extra=_parse_extra_json(args.extra_json),
            overwrite=args.overwrite,
        )
    else:
        result = validate_privileged_cache(
            cache_path=args.cache,
            dataset_index=args.dataset_index,
            teacher_provenance=args.teacher_provenance,
            expected_manifest_sha256=args.manifest_sha256,
            expected_part_names=args.part_names,
            require_exact_index_file=args.require_exact_index_file,
        )
    print(json.dumps(result.summary(), sort_keys=True))
    return 0


__all__ = ("main",)


if __name__ == "__main__":
    raise SystemExit(main())
