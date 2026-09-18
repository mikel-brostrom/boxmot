"""Click adapter for GEPA-backed tracker research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from boxmot.engine.commands._options import (
    build_selection_options,
    data_root_option,
    experiment_option,
    replay_options,
)
from boxmot.engine.commands._support import _dispatch_cli_workflow, _require_experiment_input
from boxmot.engine.config.runtime import BOXMOT_DEFAULTS


def _research_options(func):
    defaults = BOXMOT_DEFAULTS.research
    options = (
        click.option(
            "--proposal-model",
            type=str,
            default=defaults.proposal_model,
            show_default=True,
            help=(
                "proposal model identifier used by GEPA reflections, e.g. openai/gpt-5.4, "
                "anthropic/claude-sonnet-4-20250514, openrouter/openai/gpt-5.4"
            ),
        ),
        click.option(
            "--proposal-api-key",
            type=str,
            default=defaults.proposal_api_key,
            help="proposal model API key; prefer shell env vars in CI but this can inject the key at runtime",
        ),
        click.option(
            "--proposal-api-key-env",
            type=str,
            default=defaults.proposal_api_key_env,
            help=(
                "environment variable name for --proposal-api-key when the provider is not inferred, "
                "e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY"
            ),
        ),
        click.option(
            "--max-metric-calls",
            type=int,
            default=defaults.max_metric_calls,
            show_default=True,
            help="maximum number of benchmark evaluations during research",
        ),
        click.option(
            "--eval-timeout",
            type=float,
            default=defaults.eval_timeout,
            show_default=True,
            help="hard timeout in seconds for each benchmark evaluation",
        ),
        click.option(
            "--keep-workspace/--no-keep-workspace",
            default=defaults.keep_workspace,
            show_default=True,
            help="preserve the temporary research workspace after the run",
        ),
        click.option(
            "--hota-penalty",
            type=float,
            default=defaults.hota_penalty,
            show_default=True,
            help="penalty multiplier for combined HOTA regression versus baseline",
        ),
        click.option(
            "--idf1-penalty",
            type=float,
            default=defaults.idf1_penalty,
            show_default=True,
            help="penalty multiplier for combined IDF1 regression versus baseline",
        ),
        click.option(
            "--mota-penalty",
            type=float,
            default=defaults.mota_penalty,
            show_default=True,
            help="penalty multiplier for combined MOTA regression versus baseline",
        ),
        click.option(
            "--hota-tolerance",
            type=float,
            default=defaults.hota_tolerance,
            show_default=True,
            help="allowed combined HOTA drop before penalties apply",
        ),
        click.option(
            "--idf1-tolerance",
            type=float,
            default=defaults.idf1_tolerance,
            show_default=True,
            help="allowed combined IDF1 drop before penalties apply",
        ),
        click.option(
            "--mota-tolerance",
            type=float,
            default=defaults.mota_tolerance,
            show_default=True,
            help="allowed combined MOTA drop before penalties apply",
        ),
    )
    for option in reversed(options):
        func = option(func)
    return func


@click.command(help="Research tracker code changes with GEPA")
@experiment_option
@build_selection_options
@data_root_option
@replay_options(mode="research")
@_research_options
@click.pass_context
def research(
    ctx: click.Context,
    experiment: str | None,
    build_ref: str,
    build_root: Path | None,
    data_root: Path | None,
    **kwargs: Any,
) -> None:
    """Research tracker changes against an immutable materialized build."""

    experiment = _require_experiment_input(experiment, "research")
    _dispatch_cli_workflow(
        ctx,
        "research",
        "boxmot.engine.research.runner",
        {
            **kwargs,
            "experiment": experiment,
            "build": build_ref,
            "build_root": build_root,
            "data_root": data_root,
            "source": None,
            "benchmark": "",
        },
    )


__all__ = ("research",)
