# Help Center

Start with the focused troubleshooting guide, then include enough environment
and command detail when reporting a problem for someone else to reproduce it.

## Common topics

- [Installation and optional extras](../getting-started/installation.md)
- [Tracking sources and formats](../modes/track.md#inference-sources)
- [Working with tracking results](../modes/track.md#working-with-results)
- [OBB shape and angle problems](../guides/troubleshooting.md#obb-tracking)
- [ReID and accelerator problems](../guides/troubleshooting.md#reid-and-acceleration)
- [Native C++ build problems](../guides/troubleshooting.md#native-c-trackers)
- [Experiment cache behavior](../guides/troubleshooting.md#experiment-workflows)

## Diagnose first

Capture these details before opening an issue:

```bash
boxmot --help
python --version
pip show boxmot
```

Also record the exact command, full traceback, operating system, accelerator,
and detector/ReID artifact names.

## Get support

If [Troubleshooting and FAQ](../guides/troubleshooting.md) does not resolve the
problem, open a [GitHub issue](https://github.com/mikel-brostrom/boxmot/issues).
For a proposed code change, use the [Contributing Guide](../contributing/index.md).
