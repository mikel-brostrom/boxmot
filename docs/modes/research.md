# Research

Use `research` when you want GEPA to propose code changes to tracker source files and score them on a benchmark.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot research \
          --benchmark mot17-ablation \
          --tracker bytetrack \
          --proposal-model openai/gpt-5.4 \
          --max-metric-calls 24
        ```

    === "Python"

        BoxMOT does not expose a first-class public `research(...)` workflow on the high-level Python facade.

        Use the CLI for the GEPA optimization loop. Use the Python API for surrounding tasks such as tracking, validation, tuning, and exporting.

## Install the research extra

```bash
uv sync --extra research
```

## Proposal models

BoxMOT expects provider-prefixed model identifiers such as:

- `openai/gpt-5.4`
- `anthropic/claude-sonnet-4-20250514`
- `openrouter/openai/gpt-5.4`

Bare OpenAI model names such as `gpt-5.4` are normalized to `openai/gpt-5.4`, but explicit prefixes are still preferred.

## Credential setup

For standard OpenAI-compatible usage:

```bash
export OPENAI_API_KEY=...
```

For Azure OpenAI or Azure AI Foundry project routing, set the base URL to the `/openai/v1` root and let LiteLLM append `/responses` itself:

```bash
export OPENAI_BASE_URL="https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1"
export OPENAI_API_KEY=...
```

## Evaluation budget and timeout

- `max_metric_calls` limits how many benchmark evaluations GEPA can spend.
- `eval_timeout` is per evaluation subprocess, not the total wall-clock runtime of the full research job.

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
