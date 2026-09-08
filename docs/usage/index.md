# CLI

BoxMOT exposes exactly these commands:

```text
track, materialize, eval, tune, research,
train-reid, eval-reid, compare-reid, export, build
```

Run `boxmot COMMAND --help` for the authoritative option set.

## Common workflows

```bash
# Track a finite video or live source
boxmot track --source video.mp4 --detector yolov8n --tracker bytetrack --save

# Publish perception once
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml

# Materialize/reuse the experiment build, then replay it
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --device mps \
  --tracker boosttrack

boxmot tune \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker bytetrack

boxmot research \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID \
  --tracker bytetrack \
  --proposal-model openai/gpt-5.4
```

For `--build`, an existing path is used directly. An ID is resolved only under
`--build-root`, whose default is `./runs/materializations` unless
`BOXMOT_BUILDS_DIR` is set. When eval or tune omits `--build`, the resolved
experiment and inputs determine a canonical build under that root. Reuse
requires matching source and semantic component fingerprints. There is no
latest-build lookup.

## Input contracts

| Command | Input contract |
| --- | --- |
| `track` | source plus explicit components |
| `materialize` | `--experiment` |
| `eval` | `--experiment`, or `--dataset` plus `--detector` or `--build` |
| `tune` | `--experiment`, or `--dataset` plus `--detector` or `--build` |
| `research` | `--experiment` and `--build` |

Materialization gets its dataset, split, geometry, components, and class map
from the experiment. To change any of those values, select or create another
experiment; the command line exposes only execution, publication, and location
controls.

Evaluation and tuning also accept `--dataset`, `--detector`, and optional `--reid` as
shorthand for a matching authored catalog experiment. Missing or ambiguous
matches require an explicit `--experiment`. See [Evaluate](../modes/eval.md)
for an example.

Raw tracking datasets default to `./datasets/mot`; pass `--data-root` explicitly
to use another location. Materialized-dataset roots resolve independently from
`--build-root`, then `BOXMOT_BUILDS_DIR`, then `./runs/materializations`.

## Command references

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
