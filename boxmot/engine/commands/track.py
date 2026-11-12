"""Click adapter for direct object tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from boxmot.engine.commands._options import (
    association_function_option,
    edgetam_option,
    mask_guidance_max_objects_option,
    mask_guidance_weights_option,
    sequence_option,
    source_option,
    track_options,
    tracker_backend_option,
    tracker_config_option,
)
from boxmot.engine.commands._support import (
    _build_cli_namespace,
    _dispatch_cli_workflow,
    _explicit_cli_keys,
    _is_option_explicit,
)
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


def _validate_prompt_options(ctx: click.Context, payload: dict[str, Any]) -> None:
    """Reject irrelevant controls before opening an interactive tracking session."""
    if payload.get("tracker") != "edgetam":
        if _is_option_explicit(ctx, "prompt_weights"):
            raise click.UsageError("--prompt-weights requires --tracker edgetam.")
        return
    if payload.get("tracker_backend") != "python" or payload.get("geometry") not in {"auto", "aabb"}:
        raise click.UsageError("EdgeTAM prompt tracking requires --tracker-backend python and AABB geometry.")
    if payload.get("per_class"):
        raise click.UsageError("EdgeTAM prompt tracking does not support --per-class.")
    unsupported = _explicit_cli_keys(ctx).intersection(
        {
            "detector",
            "reid",
            "segmentor",
            "classes",
            "conf",
            "iou",
            "imgsz",
            "half",
            "agnostic_nms",
            "asso_func",
            "variable_dt",
            "save_crop",
            "show_kf_preds",
            "edgetam",
            "mask_guidance_weights",
            "mask_guidance_max_objects",
        }
    )
    if unsupported:
        flags = ", ".join("--" + name.replace("_", "-") for name in sorted(unsupported))
        raise click.UsageError(f"EdgeTAM uses manual box or point prompts and does not use {flags}.")


def _track_saved_detections(ctx: click.Context, payload: dict[str, Any]) -> None:
    """Replay saved TrackR-CNN predictions using the shared track command."""
    required = ("detections", "images", "instances")
    missing = [f"--{name}" for name in required if payload.get(name) is None]
    if missing:
        raise click.UsageError(f"Saved TrackR-CNN tracking requires {', '.join(missing)}.")

    supported = {
        *required,
        "split",
        "sequence_names",
        "project",
        "tracker",
        "tracker_backend",
        "tracker_config",
        "asso_func",
        "variable_dt",
        "per_class",
    }
    unsupported = _explicit_cli_keys(ctx) - supported
    if unsupported:
        options = sorted(
            option.opts[0]
            for option in ctx.command.params
            if isinstance(option, click.Option) and option.name in unsupported
        )
        raise click.UsageError(f"Saved TrackR-CNN tracking does not support {', '.join(options)}.")
    if _is_option_explicit(ctx, "tracker_backend") and payload["tracker_backend"] != "python":
        raise click.UsageError("Saved TrackR-CNN tracking requires --tracker-backend python.")

    resolved = {name: value for name, value in payload.items() if name in supported}
    if not _is_option_explicit(ctx, "tracker"):
        resolved["tracker"] = "maf_hda"
    if not _is_option_explicit(ctx, "project"):
        resolved["project"] = Path("runs/trackrcnn")
    resolved.update(tracker_backend="python", geometry="aabb", per_class=True, workflow_mode="track")
    try:
        from boxmot.engine.eval.trackrcnn import run_trackrcnn

        output = run_trackrcnn(_build_cli_namespace(ctx, "track", resolved))
    except (TypeError, ValueError, FileNotFoundError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Results: {output}")


@click.command(help="Run tracking on a source or saved TrackR-CNN predictions.")
@source_option(default=BOXMOT_DEFAULTS.track.source, help_text="file/dir/URL/glob, 0 for webcam")
@tracker_backend_option(default=BOXMOT_DEFAULTS.track.tracker_backend)
@tracker_config_option
@association_function_option
@track_options
@_singular_model_options
@click.option("--segmentor", type=str, default=None, help="Segmentor config ID or YAML path.")
@mask_guidance_weights_option
@edgetam_option
@mask_guidance_max_objects_option
@click.option(
    "--prompt-weights",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("edgetam.pt"),
    show_default=True,
    help="EdgeTAM checkpoint for --tracker edgetam; draw boxes or click points in its video window.",
)
@click.option(
    "--geometry",
    type=click.Choice(("auto", "aabb", "obb")),
    default=BOXMOT_DEFAULTS.track.geometry,
    show_default=True,
    help="Infer box geometry from the detector; aabb/obb require matching detector output.",
)
@click.option("--save-json", is_flag=True, default=False, help="Write canonical JSON Lines track output.")
@click.option(
    "--attributes", type=str, default=None, metavar="ID_OR_YAML",
    help="Attribute profile; analyze tracks in the background and write attributes.jsonl.",
)
@click.option(
    "--attributes-host", type=str, default=None, metavar="URL",
    help="Override the attribute server URL; requires --attributes.",
)
@click.option(
    "--detections",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="TrackR-CNN detection directory; requires --images and --instances. Defaults to maf_hda and runs/trackrcnn.",
)
@click.option(
    "--images",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="KITTI training/image_02 directory for saved TrackR-CNN tracking.",
)
@click.option(
    "--instances",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="KITTI MOTS ground-truth instance PNG directory for saved TrackR-CNN mask evaluation.",
)
@click.option(
    "--split",
    type=click.Choice(["val", "train", "fulltrain"]),
    default="val",
    show_default=True,
    help="KITTI MOTS split for saved TrackR-CNN tracking.",
)
@sequence_option
@click.pass_context
def track(
    ctx: click.Context,
    detector: str,
    reid: str,
    classes: str | None,
    segmentor: str | None,
    geometry: str,
    save_json: bool,
    attributes: str | None,
    attributes_host: str | None,
    detections: Path | None,
    images: Path | None,
    instances: Path | None,
    split: str,
    sequence_names: tuple[str, ...],
    **kwargs: Any,
) -> None:
    """Run the direct tracking workflow."""

    if detections is not None:
        _track_saved_detections(
            ctx,
            {
                **kwargs,
                "detections": detections,
                "images": images,
                "instances": instances,
                "split": split,
                "sequence_names": sequence_names,
            },
        )
        return
    if any(_is_option_explicit(ctx, name) for name in ("images", "instances", "split", "sequence_names")):
        raise click.UsageError("--images, --instances, --split, and --sequence require --detections.")
    if attributes_host is not None and attributes is None:
        raise click.UsageError("--attributes-host requires --attributes.")

    _validate_prompt_options(ctx, {**kwargs, "geometry": geometry})
    if kwargs.get("tracker") == "edgetam":
        detector = reid = segmentor = None
        kwargs["show"] = True
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
                "attributes": attributes,
                "attributes_host": attributes_host,
                "classes": classes,
                "source": source,
                "workflow_mode": "track",
            },
        ),
    )


__all__ = ("track",)
