# Research

Use `research` when you want GEPA to propose code changes to tracker source files and score them on a benchmark.

Reference material:

- [GEPA repository](https://github.com/gepa-ai/gepa)
- [Paper: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot research \
          --benchmark mot17 --split ablation \
          --tracker bytetrack \
          --proposal-model openai/gpt-5.4 \
          --max-metric-calls 24
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        result = BoxMOT(tracker="bytetrack").research(
            benchmark="mot17",
            split="ablation",
            proposal_model="openai/gpt-5.4",
            max_metric_calls=24,
        )
        print(result.delta_summary)
        ```

## Prerequisites

See [Mode-specific extras](../getting-started/installation.md#mode-specific-extras).

`research` needs the `research` extra for GEPA, plus whatever detector backend the selected benchmark uses.

## Proposal models

BoxMOT expects provider-prefixed model identifiers such as:

- `openai/gpt-5.4`
- `anthropic/claude-sonnet-4-20250514`
- `openrouter/openai/gpt-5.4`

Bare OpenAI model names such as `gpt-5.4` are normalized to `openai/gpt-5.4`, but explicit prefixes are still preferred.

## Credential setup

Set the provider API key in the matching environment variable, for example:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Evaluation budget and timeout

- `max_metric_calls` limits how many benchmark evaluations GEPA can spend.
- `eval_timeout` is per evaluation subprocess, not the total wall-clock runtime of the full research job.

## Native C++ scoring

Use `--tracker-backend cpp` to score candidate changes through a native C++ replay backend:

```bash
boxmot research --benchmark mot17 --split ablation --tracker bytetrack --tracker-backend cpp
```

This is only useful for trackers with registered native replay support: `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

## Outputs

`research` writes:

- GEPA state and logs
- accepted and rejected candidate artifacts
- best-candidate code snapshots
- benchmark summaries before and after optimization

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: research
    :style: table
    :prog_name: boxmot research
