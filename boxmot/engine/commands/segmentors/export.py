"""CLI adapter for the EdgeTAM TFLite component exporter."""

from __future__ import annotations

from pathlib import Path

import click

from boxmot.engine.commands._options import _parse_int_tuple


def _parse_validation_objects(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> tuple[int, ...] | None:
    """Parse positive object counts while keeping automatic validation as default."""

    if value is None:
        return None
    try:
        counts = _parse_int_tuple(ctx, param, value)
    except (TypeError, ValueError) as exc:
        raise click.BadParameter("must be a comma-separated list of positive integers") from exc
    if not counts or any(count < 1 for count in counts):
        raise click.BadParameter("must contain at least one positive object count")
    return counts


@click.command(name="export-edgetam", help="Export the full EdgeTAM TFLite bundle with a dynamic object count.")
@click.option(
    "--weights",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("edgetam.pt"),
    show_default=True,
    help="Official full EdgeTAM checkpoint; edgetam.pt downloads the default weights if absent.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory for the TFLite components, bundle manifest, and validation report.",
)
@click.option(
    "--max-objects",
    type=click.IntRange(min=1),
    default=96,
    show_default=True,
    help="Maximum object batch size to export and validate; the shared image input stays fixed.",
)
@click.option(
    "--validate-objects",
    type=str,
    callback=_parse_validation_objects,
    default=None,
    help="Object counts to test, e.g. 1,2,8,96; by default the exporter selects counts up to --max-objects.",
)
@click.option(
    "--converter-python",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Python executable with onnx2tf installed; defaults to the current interpreter.",
)
def export_edgetam(
    weights: Path,
    output: Path,
    max_objects: int,
    validate_objects: tuple[int, ...] | None,
    converter_python: Path | None,
) -> None:
    """Run reusable segmentation export only after Click validates the inputs."""

    if validate_objects is not None and max(validate_objects) > max_objects:
        raise click.BadParameter("object counts cannot exceed --max-objects", param_hint="--validate-objects")

    # Conversion dependencies remain optional and must not load for CLI help.
    from boxmot.segmentors.exporters.edgetam.export import export_edgetam as run_export

    try:
        bundle = run_export(
            weights,
            output,
            max_objects=max_objects,
            validate_objects=validate_objects,
            converter_python=converter_python,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Exported EdgeTAM TFLite bundle: {bundle}")
