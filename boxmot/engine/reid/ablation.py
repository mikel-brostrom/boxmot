"""Validated, fingerprinted orchestration helpers for ReID ablations.

The shell scripts remain convenient experiment manifests, while this module is
the single source of truth for CLI resolution, architecture validation, and
safe complete/resume/fresh run discovery.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import click
import torch
from click.core import ParameterSource

from boxmot.configs import build_mode_namespace
from boxmot.engine.cli import boxmot
from boxmot.engine.reid.data import resolve_reid_train_data
from boxmot.reid.training.config import (
    ReIDTrainConfig,
    flatten_train_hparams,
    trainer_kwargs_from_args,
)
from boxmot.reid.training.resume import (
    build_resume_contract,
    contract_differences,
    contract_fingerprint,
    run_fingerprint,
)
from boxmot.reid.training.trainer import ReIDTrainer


@dataclass(frozen=True)
class AblationSpec:
    """One fully resolved training specification."""

    trainer: ReIDTrainer
    contract: dict[str, Any]
    fingerprint: str
    run_fingerprint: str


@dataclass(frozen=True)
class RunCandidate:
    """Compatibility and progress state for one on-disk run directory."""

    path: Path
    compatible: bool
    epoch: int = 0
    complete: bool = False
    resumable: bool = False
    reason: str = ""


def _strip_remainder_separator(arguments: Sequence[str]) -> list[str]:
    args = list(arguments)
    return args[1:] if args and args[0] == "--" else args


def resolve_ablation_spec(arguments: Sequence[str]) -> AblationSpec:
    """Resolve train CLI arguments exactly as the public command does."""
    train_args = _strip_remainder_separator(arguments)
    command = boxmot.get_command(click.Context(boxmot), "train")
    if command is None:
        raise RuntimeError("The BoxMOT train command is not registered")
    context = command.make_context("train", train_args)
    explicit_keys = {
        parameter.name
        for parameter in command.params
        if isinstance(parameter, click.Option)
        and context.get_parameter_source(parameter.name) != ParameterSource.DEFAULT
    }
    namespace = build_mode_namespace("train", context.params, explicit_keys=explicit_keys)
    namespace = resolve_reid_train_data(namespace)
    if getattr(namespace, "resume", None):
        raise ValueError("A planned ablation specification must not include --resume")

    kwargs = trainer_kwargs_from_args(namespace, {})
    trainer = ReIDTrainer.from_config(ReIDTrainConfig.from_flat_kwargs(**kwargs))
    recipe = trainer._resolve_training_recipe_for_model_name()
    if recipe is not None:
        recipe.apply_pre_build_defaults(trainer)
        recipe.apply_defaults(trainer)
    trainer._validate_config()

    contract = trainer._resume_contract()
    return AblationSpec(
        trainer=trainer,
        contract=contract,
        fingerprint=contract_fingerprint(contract),
        run_fingerprint=run_fingerprint(contract, trainer.epochs),
    )


def validate_ablation_spec(spec: AblationSpec, *, forward: bool = True) -> int:
    """Build the real architecture and optionally execute a CPU dummy forward."""
    trainer = spec.trainer
    pretrained = trainer.pretrained
    device = trainer.device
    try:
        # Validation must be deterministic, offline, and independent of the
        # selected training accelerator. Pretraining itself is in the contract.
        trainer.pretrained = False
        trainer.device = torch.device("cpu")
        model = trainer._build_model(num_classes=8).cpu().eval()
        if not forward:
            return sum(parameter.numel() for parameter in model.parameters())
        sample = torch.zeros(1, 3, *trainer.img_size)
        with torch.inference_mode():
            output = model(sample)
        if not torch.is_tensor(output) or output.ndim != 2 or output.shape[0] != 1:
            raise RuntimeError(
                "Validation forward must return one two-dimensional retrieval descriptor; "
                f"got {type(output).__name__} with shape={getattr(output, 'shape', None)}"
            )
        if output.shape[1] < 1 or not torch.isfinite(output).all():
            raise RuntimeError("Validation forward returned an empty or non-finite descriptor")
        return int(output.shape[1])
    finally:
        trainer.pretrained = pretrained
        trainer.device = device


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_metrics_epoch(metrics: dict[str, Any]) -> int:
    train = metrics.get("train") or []
    return max((int(item.get("epoch", 0)) for item in train), default=0)


def _candidate_directories(project: Path, name: str) -> list[Path]:
    candidates = [project / name]
    suffixes = []
    for candidate in project.glob(f"{name}_[0-9]*"):
        suffix = candidate.name.removeprefix(f"{name}_")
        if candidate.is_dir() and suffix.isdigit():
            suffixes.append((int(suffix), candidate))
    candidates.extend(path for _, path in sorted(suffixes))
    return [candidate for candidate in candidates if candidate.is_dir()]


def _saved_contract(hparams: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    resume = hparams.get("resume")
    if isinstance(resume, dict) and isinstance(resume.get("contract"), dict):
        return resume["contract"], False
    return build_resume_contract(flatten_train_hparams(hparams), partial=True), True


def _inspect_candidate(path: Path, spec: AblationSpec) -> RunCandidate:
    hparams_path = path / "hparams.json"
    if not hparams_path.exists():
        return RunCandidate(path, False, reason="missing hparams.json")
    try:
        hparams = _read_json(hparams_path)
        saved_contract, legacy = _saved_contract(hparams)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return RunCandidate(path, False, reason=f"invalid hparams.json: {exc}")

    differences = contract_differences(
        saved_contract,
        spec.contract,
        compare_common_only=legacy,
    )
    if differences:
        return RunCandidate(path, False, reason="; ".join(differences[:3]))

    metrics_path = path / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = _read_json(metrics_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return RunCandidate(path, False, reason=f"invalid metrics.json: {exc}")
    metrics_epoch = _last_metrics_epoch(metrics)
    last_path = path / "last.pt"
    if metrics_epoch >= spec.trainer.epochs:
        return RunCandidate(path, True, epoch=metrics_epoch, complete=True)

    if not last_path.is_file():
        return RunCandidate(
            path,
            False,
            epoch=metrics_epoch,
            reason="incomplete run has no last.pt",
        )
    try:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return RunCandidate(path, False, epoch=metrics_epoch, reason=f"invalid last.pt: {exc}")
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    if not checkpoint.get("resumable", "optimizer" in checkpoint):
        return RunCandidate(path, False, epoch=checkpoint_epoch, reason="last.pt is not resumable")
    checkpoint_contract = checkpoint.get("resume_contract")
    if checkpoint_contract is not None:
        checkpoint_differences = contract_differences(checkpoint_contract, spec.contract)
        if checkpoint_differences:
            return RunCandidate(
                path,
                False,
                epoch=checkpoint_epoch,
                reason="checkpoint mismatch: " + "; ".join(checkpoint_differences[:3]),
            )
    if checkpoint_epoch >= spec.trainer.epochs:
        return RunCandidate(
            path,
            False,
            epoch=checkpoint_epoch,
            reason="checkpoint reached target but completed metrics/checkpoints are inconsistent",
        )
    required = {"optimizer", "optimizer_center", "rng_state"}
    if spec.trainer.center_loss_weight > 0:
        required.add("center_loss_state_dict")
    if spec.trainer.classifier_loss != "ce":
        required.add("classifier_loss_state_dict")
    if spec.trainer.ema_decay:
        required.add("ema_state_dict")
    if checkpoint_contract is not None:
        required.add("scheduler")
    if spec.trainer.deterministic and spec.trainer.device.type == "cuda":
        required.add("grad_scaler")
    missing = sorted(required - checkpoint.keys())
    if missing:
        return RunCandidate(
            path,
            False,
            epoch=checkpoint_epoch,
            reason="last.pt missing " + ", ".join(missing),
        )
    if metrics_path.exists() and metrics_epoch < checkpoint_epoch:
        return RunCandidate(
            path,
            False,
            epoch=checkpoint_epoch,
            reason=f"checkpoint epoch {checkpoint_epoch} is ahead of metrics epoch {metrics_epoch}",
        )
    return RunCandidate(path, True, epoch=checkpoint_epoch, resumable=True)


def discover_run(project: Path, name: str, spec: AblationSpec) -> tuple[str, Path, str]:
    """Return complete, resume, fresh, or incompatible for one experiment."""
    candidates = _candidate_directories(project, name)
    if not candidates:
        return "fresh", project / name, "no existing run"
    inspected = [_inspect_candidate(candidate, spec) for candidate in candidates]
    completed = [candidate for candidate in inspected if candidate.complete]
    if completed:
        selected = max(completed, key=lambda candidate: (candidate.epoch, str(candidate.path)))
        return "complete", selected.path, f"epoch {selected.epoch}"
    resumable = [candidate for candidate in inspected if candidate.resumable]
    if resumable:
        selected = max(resumable, key=lambda candidate: (candidate.epoch, str(candidate.path)))
        return "resume", selected.path / "last.pt", f"epoch {selected.epoch}"
    reasons = " | ".join(f"{item.path}: {item.reason}" for item in inspected)
    return "incompatible", inspected[-1].path, reasons


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
    """CLI entry point used by ablation shell manifests."""
    args = _build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
