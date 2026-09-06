# Research

`research` proposes tracker code changes and scores them against a fixed
experiment and immutable build. It requires both inputs:

```bash
boxmot research \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID \
  --tracker bytetrack \
  --proposal-model openai/gpt-5.4 \
  --max-metric-calls 24
```

The parent process validates the build before creating isolated workspaces.
Each candidate replays selected keyed sequences through the live tracker API;
candidate workspaces do not perform perception inference or mutate the build.

Run materialization separately when required:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
```

There is no implicit build creation or latest-build lookup. This makes metric
changes attributable to tracker code or parameters rather than changing
detections and embeddings.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.research
    :command: research
    :prog_name: boxmot research
    :depth: 0
