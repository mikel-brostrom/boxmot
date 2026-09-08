#!/usr/bin/env python3
"""BoxMOT command-line entrypoint and lazy command registry."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import click

from boxmot import __version__
from boxmot._tracker_exports import _TRACKER_MANIFEST


@dataclass(frozen=True, slots=True)
class _CommandSpec:
    """Import metadata and root-help text for one CLI command."""

    name: str
    module: str
    attribute: str
    summary: str


_COMMAND_SPECS = (
    _CommandSpec("track", "boxmot.engine.commands.track", "track", "Track objects in video/webcam stream"),
    _CommandSpec(
        "materialize",
        "boxmot.engine.commands.materialize",
        "materialize",
        "Build an immutable keyed perception dataset",
    ),
    _CommandSpec(
        "time-variant",
        "boxmot.engine.commands.time_variant",
        "time_variant",
        "Derive a timestamped frame-loss dataset from a build",
    ),
    _CommandSpec("eval", "boxmot.engine.commands.eval", "eval", "Evaluate tracker performance on MOT dataset"),
    _CommandSpec("tune", "boxmot.engine.commands.tune", "tune", "Optimize tracker hyperparameters"),
    _CommandSpec(
        "research",
        "boxmot.engine.commands.research",
        "research",
        "Evolve tracker code against benchmark metrics",
    ),
    _CommandSpec(
        "train-reid",
        "boxmot.engine.commands.reid.train",
        "train_reid",
        "Train a ReID model on a person/vehicle dataset",
    ),
    _CommandSpec(
        "eval-reid",
        "boxmot.engine.commands.reid.evaluate",
        "eval_reid",
        "Evaluate a trained ReID model on query/gallery data",
    ),
    _CommandSpec(
        "compare-reid",
        "boxmot.engine.commands.reid.compare",
        "compare_reid",
        "Compare ReID checkpoints across target datasets",
    ),
    _CommandSpec("export", "boxmot.engine.commands.reid.export", "export", "Export ReID models to different formats"),
    _CommandSpec("build", "boxmot.engine.commands.build", "build", "Build native tracker extensions"),
)
_COMMAND_SPEC_BY_NAME = {spec.name: spec for spec in _COMMAND_SPECS}
_TRACKER_HELP = ", ".join(_TRACKER_MANIFEST)


class CommandFirstGroup(click.Group):
    """Click group with BoxMOT help formatting and lazy child resolution."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List every registered command without importing its adapter module."""

        del ctx
        return [spec.name for spec in _COMMAND_SPECS]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Import and cache only the command selected by Click."""

        if command := self.commands.get(cmd_name):
            return command
        spec = _COMMAND_SPEC_BY_NAME.get(cmd_name)
        if spec is None:
            return None

        module = importlib.import_module(spec.module)
        command = getattr(module, spec.attribute)
        if not isinstance(command, click.Command):
            raise TypeError(f"{spec.module}.{spec.attribute} is not a Click command")
        if command.name != spec.name:
            raise ValueError(
                f"CLI manifest registers {spec.module}.{spec.attribute} as {spec.name!r}, "
                f"but the command declares {command.name!r}"
            )
        self.add_command(command)
        return command

    def format_help(self, _ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Render the command-first BoxMOT help without resolving child commands."""

        formatter.write_paragraph()
        formatter.write_text("BoxMOT 'boxmot' commands use the following syntax:")
        formatter.write_paragraph()

        with formatter.indentation():
            formatter.write_text("boxmot MODE [OPTIONS]")
        formatter.write_paragraph()

        formatter.width = 120
        with formatter.indentation():
            modes = ", ".join(spec.name for spec in _COMMAND_SPECS)
            formatter.write_text(f"Where  MODE (required) is one of [{modes}]")
            formatter.write_text("       --detector selects a YOLO model like yolov8n, yolov9c, yolo11m, yolox_x")
            formatter.write_text("       --reid selects a ReID model like osnet_x0_25_msmt17, mobilenetv2_x1_4")
            formatter.write_text(f"       --tracker selects one of [{_TRACKER_HELP}]")
            formatter.write_text(
                "       OPTIONS (optional) flags like '--source 0' for tracking inputs or "
                "'--dataset mot17 --split ablation' for model-free evaluation."
            )
            formatter.write_text(
                "       --experiment fixes the dataset, split, geometry, detector, optional segmentor and ReID, "
                "and class map."
            )
            formatter.write_text(
                "          See all options at https://github.com/mikel-brostrom/boxmot or 'boxmot MODE --help'"
            )
        formatter.write_paragraph()

        formatter.write_text("Examples:")
        with formatter.indentation():
            formatter.write_text("1. Track with webcam using defaults:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker deepocsort --source 0 --show"
                )
            formatter.write_paragraph()

            formatter.write_text("2. Track a video file:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 "
                    "--tracker botsort --source video.mp4 --save"
                )
            formatter.write_paragraph()

            formatter.write_text("3. Evaluate on MOT dataset:")
            with formatter.indentation():
                formatter.write_text("boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --tracker boosttrack")
            formatter.write_paragraph()

            formatter.write_text("4. Tune tracker hyperparameters:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID "
                    "--tracker deepocsort --n-trials 10"
                )
            formatter.write_paragraph()

            formatter.write_text("5. Research tracker code changes:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot research --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID "
                    "--tracker bytetrack "
                    "--proposal-model openai/gpt-5.4 --max-metric-calls 24"
                )
            formatter.write_paragraph()

            formatter.write_text("6. Train a ReID model:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot train-reid --model osnet_x0_25 --dataset market1501 "
                    "--data-dir /path/to/data --epochs 120 --device 0"
                )
            formatter.write_paragraph()

            formatter.write_text("7. Train on all person datasets jointly:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot train-reid --model csl_tinyvit_11m --dataset market1501,duke,cuhk03,msmt17 "
                    "--data-dir /path/to/data --device 0"
                )
            formatter.write_paragraph()

            formatter.write_text("8. Export ReID model:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic"
                )
        formatter.write_paragraph()

        formatter.write_text("Modes:")
        with formatter.indentation():
            width = max(len(spec.name) for spec in _COMMAND_SPECS)
            for spec in _COMMAND_SPECS:
                formatter.write_text(f"{spec.name:<{width}} {spec.summary}")
        formatter.write_paragraph()

        formatter.write_text("Docs:      https://github.com/mikel-brostrom/boxmot")
        formatter.write_text("Community: https://github.com/mikel-brostrom/boxmot/discussions")


@click.group(cls=CommandFirstGroup)
@click.version_option(__version__, prog_name="BoxMOT")
@click.pass_context
def boxmot(ctx: click.Context) -> None:
    """Pluggable object tracking for detection, segmentation, and pose models."""

    del ctx
    from boxmot.engine.logging import configure_engine_logging

    configure_engine_logging()


main = boxmot

__all__ = ("boxmot", "main")


if __name__ == "__main__":
    boxmot()
