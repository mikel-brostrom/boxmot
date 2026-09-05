"""Click adapter for native BoxMOT library builds."""

from __future__ import annotations

import importlib

import click


@click.command(help="Build native (C++) tracker shared libraries")
@click.option(
    "--tracker",
    "trackers",
    multiple=True,
    type=click.Choice(["all", "botsort", "bytetrack", "occluboost", "ocsort", "sfsort", "reid"]),
    default=("all",),
    help='Tracker(s) to build. Pass --tracker multiple times or use "all" (default).',
)
@click.option("--force", is_flag=True, default=False, help="Force rebuild even if libraries already exist.")
def build(trackers: tuple[str, ...], force: bool) -> None:
    """Compile the native C++ shared libraries shipped with BoxMOT."""

    selected = set(trackers)
    if "all" in selected:
        selected = {"reid", "botsort", "bytetrack", "occluboost", "ocsort", "sfsort"}

    order = ["reid", "botsort", "bytetrack", "occluboost", "ocsort", "sfsort"]
    selected_in_order = [name for name in order if name in selected]

    failures: list[tuple[str, str]] = []
    for name in selected_in_order:
        try:
            if name == "reid":
                from boxmot.native.reid import ensure_reid_capi_library

                library = ensure_reid_capi_library(force_rebuild=force)
            else:
                module = importlib.import_module(f"boxmot.native.trackers.{name}")
                ensure = getattr(module, f"ensure_{name}_cpp_library")
                library = ensure(force_rebuild=force)
            click.echo(f"[boxmot build] {name}: built -> {library}")
        except Exception as exc:  # noqa: BLE001 - surface CMake errors verbatim
            failures.append((name, str(exc)))
            click.echo(f"[boxmot build] {name}: FAILED\n{exc}", err=True)

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise click.ClickException(f"Native build failed for: {names}")


__all__ = ("build",)
