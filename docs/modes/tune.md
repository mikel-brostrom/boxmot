# Tune

`tune` optimizes tracker parameters while replaying the same immutable
perception build for every trial. It requires an experiment and an explicit
build:

```bash
boxmot tune \
  --experiment mot17-ablation-yolox-lmbn \
  --build BUILD_ID \
  --tracker bytetrack \
  --n-trials 50
```

Materialize first if the build does not exist:

```bash
boxmot materialize --experiment mot17-ablation-yolox-lmbn
```

Tuning validates source, split, taxonomy, geometry, component fingerprints,
and the selected tracker's requirements once before optimization. Trials run no
detector, segmentor, or encoder and cannot silently select or create another
build. Worker count and retry policy are execution settings, not semantic
fingerprints.

Use a native tracker with `--tracker-backend cpp` when that geometry and feature
combination is supported. Unsupported masks, per-class mode, or geometry are
rejected before a native trial starts.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.tune
    :command: tune
    :prog_name: boxmot tune
    :depth: 0
