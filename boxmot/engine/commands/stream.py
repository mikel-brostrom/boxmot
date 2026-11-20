"""CLI adapters for continuous cameras and explicit attribute database upkeep."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Continuous camera YAML configuration.",
)
def stream(config_path: Path) -> None:
    """Supervise a headless network camera and retain JSON metadata."""
    from boxmot.engine.config.streams import load_stream_config
    from boxmot.engine.streaming.runtime import run_stream
    from boxmot.utils.config import ConfigurationError

    try:
        run_stream(load_stream_config(config_path))
    except ConfigurationError as exc:
        raise click.ClickException(str(exc)) from None
    except Exception as exc:
        # Drivers and native decoders can include credentialed URLs in errors.
        raise click.ClickException(
            f"Stream stopped ({type(exc).__name__}). Check health, configuration and dependencies."
        ) from None


@click.group("stream-db")
def stream_db() -> None:
    """Upgrade attribute storage or clean expired continuous-camera history."""


@stream_db.command()
@click.option("--database-url-env", default="BOXMOT_ATTRIBUTES_DATABASE_URL", show_default=True)
def upgrade(database_url_env: str) -> None:
    """Apply the additive schema upgrade with existing clients stopped."""
    from boxmot.attributes.database import upgrade_database

    upgrade_database(database_url_env)
    click.echo("Attribute schema is current.")


@stream_db.command()
@click.option("--database-url-env", default="BOXMOT_ATTRIBUTES_DATABASE_URL", show_default=True)
@click.option("--retention-days", type=click.IntRange(min=1), default=7, show_default=True)
def maintain(database_url_env: str, retention_days: int) -> None:
    """Cancel orphan work and prune acknowledged historical database records."""
    from boxmot.attributes.streams import maintain_database

    maintain_database(database_url_env, retention_days)


@click.group("stream-outputs")
def stream_outputs() -> None:
    """Maintain one camera's exported metadata without global database cleanup."""


@stream_outputs.command("maintain")
@click.option("--database-url-env", default="BOXMOT_ATTRIBUTES_DATABASE_URL", show_default=True)
@click.option("--retention-days", type=click.IntRange(min=1), default=7, show_default=True)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), required=True, help="Root containing camera directories."
)
@click.option("--camera-id", required=True, help="Stable camera identity to maintain.")
def maintain_camera_outputs(database_url_env: str, retention_days: int, output_dir: Path, camera_id: str) -> None:
    """Export abandoned events and prune this camera's expired output segments."""
    from boxmot.engine.config.streams import StreamConfig
    from boxmot.engine.streaming.maintenance import maintain_outputs
    from boxmot.utils.config import ConfigurationError

    try:
        StreamConfig(camera_id=camera_id)
    except ConfigurationError as exc:
        raise click.BadParameter(str(exc), param_hint="--camera-id") from None
    maintain_outputs(database_url_env, camera_id, output_dir, retention_days)
