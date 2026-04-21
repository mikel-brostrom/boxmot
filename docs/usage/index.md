# Usage

BoxMOT supports three main entry points:

| Entry point | Use it when | Start here |
| --- | --- | --- |
| CLI | You want one command per workflow from the terminal | [CLI](cli.md) |
| Python | You want BoxMOT embedded in an application or notebook | [Python](python.md) |
| YAML configs | You want repeatable benchmark workflows and shared defaults | [Configuration](configuration.md) |

## What stays shared

The CLI and Python facade both resolve defaults from the same config system under `boxmot/configs`, so detector, ReID, tracker, and benchmark defaults stay aligned across interfaces.

## Typical progression

1. Start with [Quickstart](../index.md).
2. Pick a workflow in [Modes Overview](../modes/index.md).
3. Use [CLI](cli.md) or [Python](python.md) depending on your integration path.
4. Move into [Configuration](configuration.md) when you need repeatable benchmark runs.
