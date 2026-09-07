# Guides

Guides connect BoxMOT's modes and configuration into complete workflows. Use
the mode pages for command arguments and the generated reference for exact
Python signatures.

## Core concepts

- [Canonical Geometry](../concepts/index.md) — AABB, OBB, confidence, class,
  and detection-index structures.
- [Tracking Pipeline](../concepts/tracking-pipeline.md) — component boundaries,
  enrichment, state, and engine ownership.

## Reproducible workflows

- [Migrating to v24](v24-migration.md) — breaking Python, CLI, cache, and native
  API changes.
- [Experiment Workflows](experiments.md) — profiles, immutable build reuse,
  public inputs, and repeatable runs.
- [Evaluation and Postprocessing](evaluation.md) — metrics, TrackEval parity,
  Kalman-filter tuning, and output transforms.
- [Compare](../compare/index.md) — choose trackers and compare them on shared
  inputs.

## Deployment and integration

- [Integrations](../integrations/index.md) — detector, ReID runtime, Python,
  service, and native extension points.
- [Tracker Service](deployment.md) — stateful HTTP tracking sessions.
- [Native C++](../native/index.md) — typed live tracking and cached evaluation
  through the same stateful API.

## Advanced ReID

- [CSL-TinyViT-7M HP-GRD](csl-tinyvit-7m-hpgrd.md) — architecture and research
  recipe details.
