"""Command adapter for validated ReID ablation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from boxmot.reid.training.ablation_runs import AblationSpec


def _strip_remainder_separator(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    return values[1:] if values and values[0] == "--" else values


def resolve_ablation_spec(arguments: Sequence[str]) -> AblationSpec:
    """Resolve train CLI arguments through the canonical ReID training configuration."""

    from boxmot.engine.commands._support import _explicit_cli_keys
    from boxmot.engine.commands.reid.train import train_reid
    from boxmot.reid.datasets.resolution import resolve_reid_train_data
    from boxmot.reid.training.ablation_runs import build_ablation_spec
    from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
    from boxmot.reid.training.presets import build_training_namespace

    context = train_reid.make_context("train-reid", _strip_remainder_separator(arguments))
    namespace = build_training_namespace(
        context.params,
        explicit_keys=_explicit_cli_keys(context),
    )
    namespace = resolve_reid_train_data(namespace)
    if getattr(namespace, "resume", None):
        raise ValueError("A planned ablation specification must not include --resume")
    kwargs = trainer_kwargs_from_args(namespace, {})
    return build_ablation_spec(ReIDTrainConfig.from_flat_kwargs(**kwargs))


def _safe_field(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\n", " ")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="resolve, build, and forward one variant")
    validate.add_argument("train_args", nargs=argparse.REMAINDER)
    discover = subparsers.add_parser("discover", help="select a compatible run action")
    discover.add_argument("--project", type=Path, required=True)
    discover.add_argument("--name", required=True)
    discover.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve and validate a shell-manifest ablation request."""

    args = _build_parser().parse_args(argv)

    from boxmot.reid.training.ablation_runs import discover_run, validate_ablation_spec

    spec = resolve_ablation_spec(args.train_args)
    if args.command == "validate":
        descriptor_dim = validate_ablation_spec(spec)
        print(
            "VALID\t{}\t{}\t{}\t{}".format(
                spec.fingerprint,
                spec.run_fingerprint,
                descriptor_dim,
                spec.trainer.train_batch_size,
            )
        )
        return 0
    action, path, reason = discover_run(args.project, args.name, spec)
    print(
        "{}\t{}\t{}\t{}\t{}".format(
            action,
            _safe_field(path),
            spec.fingerprint,
            spec.run_fingerprint,
            _safe_field(reason),
        )
    )
    return 0


__all__ = ("main", "resolve_ablation_spec")


if __name__ == "__main__":
    raise SystemExit(main())
