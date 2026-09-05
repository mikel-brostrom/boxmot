# Reference

Use the reference section when you know which interface you need and want its
exact arguments, result types, or source documentation. Start with
[Modes](../modes/index.md) for workflow guidance.

## Command line

The [CLI reference](../usage/index.md) explains the command grammar and links
to the generated option table on every mode page.

## Python

- [Python API Guide](../python/index.md) — canonical structures, immutable
  specs, component factories, pipelines, and cached dataset loading.
- [Public API](../python/high-level.md) — generated documentation for supported
  structures, pipelines, and dataset values.
- [Component API](../python/low-level.md) — generated documentation for
  detector, segmentor, appearance-encoder, and tracker contracts and factories.

## Full source reference

The [Full Source Reference](../reference/index.md) mirrors the non-entry-point
Python module hierarchy under `boxmot` and is regenerated from signatures and
docstrings during every documentation build. It includes public symbols from
internal implementation modules for contributors and advanced integrations;
private names and imported re-exports are omitted.

Use the curated Public and Component API pages for supported entry points. The
full source reference is broader, but its internal interfaces can change as the
implementation is refactored.
