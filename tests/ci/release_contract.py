"""Shared installed-package release checks, runnable without the source package."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import subprocess

EXPECTED_PUBLIC_API = (
    "__version__",
    "create_tracker",
    "BoostTrack",
    "BotSort",
    "ByteTrack",
    "DeepOcSort",
    "HybridSort",
    "OccluBoost",
    "OcSort",
    "Sam2Mot",
    "SFSORT",
    "StrongSort",
)
EXPECTED_CLI_COMMANDS = (
    "track",
    "materialize",
    "time-variant",
    "eval",
    "tune",
    "research",
    "train-reid",
    "eval-reid",
    "compare-reid",
    "export",
    "build",
)


def check_release_contract(expected_version: str | None = None) -> None:
    """Check the requested release or installed version and public discovery APIs.

    Explicit release versions also support source-only service images without
    distribution metadata. Otherwise, the installed distribution is the version
    authority; editable installs must be refreshed after source version changes.
    """
    import click

    import boxmot
    from boxmot.engine.cli import boxmot as boxmot_cli
    from boxmot.engine.experiment_config import resolve_experiment_config

    if expected_version is None:
        expected_version = importlib.metadata.version("boxmot")
    assert boxmot.__version__ == expected_version, (
        f"Package version: expected {expected_version!r}, got {boxmot.__version__!r}"
    )
    assert boxmot.__all__ == EXPECTED_PUBLIC_API, (
        f"Public API: expected {EXPECTED_PUBLIC_API!r}, got {boxmot.__all__!r}"
    )
    with click.Context(boxmot_cli) as context:
        commands = tuple(boxmot_cli.list_commands(context))
    assert commands == EXPECTED_CLI_COMMANDS, (
        f"CLI commands: expected {EXPECTED_CLI_COMMANDS!r}, got {commands!r}"
    )
    assert importlib.util.find_spec("boxmot.api") is None
    assert importlib.util.find_spec("boxmot.data") is None
    for name in ("BoxMOT", "Detector", "ReIDModel"):
        assert not hasattr(boxmot, name), f"Removed public alias remains: {name}"
    experiment = resolve_experiment_config("mot17/ablation-yolox-lmbn.yaml")
    assert experiment["detector"]["id"] == "yolox-x-mot17"
    assert experiment["reid"]["id"] == "lmbn-n-duke"


def check_cli_help() -> None:
    """Resolve each advertised command through the installed console entrypoint."""
    for command in EXPECTED_CLI_COMMANDS:
        result = subprocess.run(["boxmot", command, "--help"], capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"boxmot {command} --help failed:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        help="Expected runtime version; defaults to the installed BoxMOT distribution version.",
    )
    parser.add_argument("--check-cli-help", action="store_true", help="Also exercise every installed command's help.")
    args = parser.parse_args()
    check_release_contract(expected_version=args.expected_version)
    if args.check_cli_help:
        check_cli_help()
    print("Release public API, CLI, and packaged configuration checks passed.")
