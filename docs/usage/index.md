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
boxmot materialize --experiment mot17-ablation-yolox-lmbn

# Replay one explicit immutable build
boxmot eval \
  --experiment mot17-ablation-yolox-lmbn \
  --build BUILD_ID \
  --tracker boosttrack

boxmot tune \
  --experiment mot17-ablation-yolox-lmbn \
  --build BUILD_ID \
  --tracker bytetrack

boxmot research \
  --experiment mot17-ablation-yolox-lmbn \
  --build BUILD_ID \
  --tracker bytetrack \
  --proposal-model openai/gpt-5.4
```

For `--build`, an existing path is used directly. An ID is resolved only under
`--build-root`, whose default comes from `BOXMOT_BUILDS_DIR` or the platform
cache. There is no implicit materialization or latest-build lookup.

## Input contracts

| Command | Input contract |
| --- | --- |
| `track` | source plus explicit components |
| `materialize` | `--experiment` |
| `eval` | exactly one of `--experiment` or `--dataset`, plus `--build` |
| `tune` | `--experiment` and `--build` |
| `research` | `--experiment` and `--build` |

Materialization gets its dataset, split, geometry, components, and class map
from the experiment. To change any of those values, select or create another
experiment; the command line exposes only execution, publication, and location
controls.

Raw dataset roots resolve from `--data-root`, then `BOXMOT_DATASETS_DIR`, then
the platform cache. Build roots resolve independently from `--build-root`, then
`BOXMOT_BUILDS_DIR`, then the platform cache.

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
