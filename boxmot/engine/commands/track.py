"""Click adapter for direct object tracking."""

from __future__ import annotations

from typing import Any

import click

from boxmot.engine.commands._options import (
    association_function_option,
    source_option,
    track_options,
    tracker_backend_option,
    tracker_config_option,
)
from boxmot.engine.commands._support import _dispatch_cli_workflow, _is_option_explicit
from boxmot.engine.config.runtime import BOXMOT_DEFAULTS


def _singular_model_options(func):
    """Attach the detector, ReID, and class selectors used by tracking."""

    options = (
        click.option(
            "--detector",
            type=str,
            default=BOXMOT_DEFAULTS.shared.detector,
            help="detector profile ID, YAML config, or artifact path",
        ),
        click.option(
            "--reid",
            type=str,
            default=BOXMOT_DEFAULTS.shared.reid,
            help="ReID profile ID, YAML config, or artifact path",
        ),
        click.option("--classes", type=str, default=None, help='filter by class indices, e.g. 0 or "0,1"'),
    )
    for option in reversed(options):
        func = option(func)
    return func


def _is_live_source_value(source: str | None) -> bool:
    if source is None:
        return False
    return str(source).isdigit() or "://" in str(source)


def _apply_track_cli_defaults(ctx: click.Context, payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    source = resolved.get("source")
    has_explicit_output = any(_is_option_explicit(ctx, option_name) for option_name in ("show", "save", "save_txt"))
    if _is_live_source_value(source) and not has_explicit_output:
        resolved["show"] = True
    return resolved


@click.command(help="Run tracking only")
@source_option(default=BOXMOT_DEFAULTS.track.source, help_text="file/dir/URL/glob, 0 for webcam")
@tracker_backend_option(default=BOXMOT_DEFAULTS.track.tracker_backend)
@tracker_config_option
@association_function_option
@track_options
@_singular_model_options
@click.option("--segmentor", type=str, default=None, help="Segmentor config ID or YAML path.")
@click.option("--geometry", type=click.Choice(("aabb", "obb")), default="aabb", show_default=True)
@click.option("--save-json", is_flag=True, default=False, help="Write canonical JSON Lines track output.")
@click.pass_context
def track(
    ctx: click.Context,
    detector: str,
    reid: str,
    classes: str | None,
    segmentor: str | None,
    geometry: str,
    save_json: bool,
    **kwargs: Any,
) -> None:
    """Run the direct tracking workflow."""

    source = kwargs.pop("source")
    _dispatch_cli_workflow(
        ctx,
        "track",
        "boxmot.engine.tracking.workflow",
        _apply_track_cli_defaults(
            ctx,
            {
                **kwargs,
                "detector": detector,
                "reid": reid,
                "segmentor": segmentor,
                "geometry": geometry,
                "save_json": save_json,
                "classes": classes,
                "source": source,
                "workflow_mode": "track",
            },
        ),
    )


__all__ = ("track",)
