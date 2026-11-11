# CLI

The main workflow and environment commands are:

```text
ui, install, track, materialize, eval, tune, research,
train-reid, eval-reid, compare-reid, export, build
```

Run `boxmot COMMAND --help` for the authoritative option set.

For paired **CLI** and **Python** examples, open the
[mode guides](../modes/index.md) and select either tab.

Use `boxmot --version` for the installed version. The global `--dashboard`
option goes **before** the command, for example
`boxmot --dashboard track --source video.mp4`. Use [the terminal UI](../modes/ui.md)
to configure a command interactively.

`export-edgetam` also appears in help, but its conversion backend is absent in
this checkout. Use the supported [PyTorch mask workflows](../tasks/masks.md)
instead of treating that draft adapter as a working exporter.

## Common workflows

The YOLOX benchmark examples require the
[source checkout](../getting-started/pytorch.md#source-checkout) with `yolo`;
add `evolve` for tuning and `research` for code search. In that checkout, prefix
the commands below with `uv run --no-sync`.

```bash
# Track a finite video or live source
boxmot track --source video.mp4 --detector yolov8n --tracker bytetrack --save

# Publish perception once
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml

# Replay the explicit build
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID \
  --tracker boosttrack

boxmot tune \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID \
  --tracker bytetrack

boxmot research \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID \
  --tracker bytetrack \
  --proposal-model openai/gpt-5.4
```

Replace `video.mp4` with your input and `BUILD_ID` with the printed build ID.
Research invokes an external proposal model; configure its credentials as
described in [Research](../modes/research.md).

For `--build`, an existing path is used directly. An ID is resolved only under
`--build-root`, whose default is `./runs/materializations` unless
`BOXMOT_BUILDS_DIR` is set. When eval or tune omits `--build`, the resolved
experiment and inputs determine a canonical build under that root. Reuse
requires matching source and semantic component fingerprints. There is no
latest-build lookup.

## Input contracts

| Command | Input contract |
| --- | --- |
| `track` | `--source` plus components, or the saved-mask replay paths |
| `materialize` | image `--experiment`, or `--time-variant` with dataset, sequence, and parent build |
| `eval` | image experiment/build, saved 2D predictions, or an EagerMOT sensor dataset |
| `tune` | image experiment/build or an EagerMOT sensor dataset; saved 2D boxes are not supported |
| `research` | `--experiment` and `--build` |

Image materialization gets its dataset, split, geometry, components, and class map
from the experiment. To change any of those values, select or create another
experiment; the command line exposes only execution, publication, and location
controls.

Evaluation and tuning also accept `--dataset`, `--detector`, and optional `--reid` as
shorthand for a matching authored catalog experiment. Missing or ambiguous
matches require an explicit `--experiment`. See [Evaluate](../modes/eval.md)
for an example. Saved prediction/sensor datasets instead supply observations
from their declared modalities; they do not use a perception build. See
[Datasets](../datasets/index.md) for the supported input paths.

Raw tracking datasets default to `./datasets/mot`; pass `--data-root` explicitly
to use another location. Materialized-dataset roots resolve independently from
`--build-root`, then `BOXMOT_BUILDS_DIR`, then `./runs/materializations`.

## Command references

- [Terminal UI](../modes/ui.md)
- [Install dependencies](../modes/install.md)
- [Track](../modes/track.md)
- [Materialize](../modes/materialize.md)
- [Eval](../modes/eval.md)
- [Tune](../modes/tune.md)
- [Research](../modes/research.md)
- [Train ReID](../modes/train.md)
- [Evaluate ReID](../modes/eval-reid.md)
- [Compare ReID](../modes/compare-reid.md)
- [Export](../modes/export.md)
- [Native build](../native/index.md)

## Global options

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 0
    :style: table
    :prog_name: boxmot
