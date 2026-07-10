# Evaluate ReID

Use `eval-reid` to score a trained ReID checkpoint on a dataset's query/gallery split.

## Examples

!!! example

    === "CLI"

        Evaluate a checkpoint on Market1501:

        ```bash
        boxmot eval-reid \
          --weights runs/reid_train/osnet_market/best.pt \
          --dataset market1501 \
          --data-dir /data/reid \
          --device 0
        ```

        Save results to a custom directory:

        ```bash
        boxmot eval-reid \
          --weights runs/reid_train/lmbn_joint/best.pt \
          --dataset msmt17 \
          --data-dir /data/reid \
          --batch-size 128 \
          --output runs/reid_eval
        ```

## What it does

`eval-reid` loads a trained checkpoint, rebuilds the matching backbone, extracts query and gallery embeddings, computes the distance matrix, and reports ranking metrics.

If the checkpoint stores the model architecture, you can omit `--model`. Otherwise pass the architecture explicitly.

## Main outputs

The command reports and saves:

- `mAP`
- `rank1`
- `rank5`
- `rank10`

By default the JSON summary is written next to the checkpoint as `eval_<dataset>.json`. Use `--output` to place it elsewhere.

## Dataset expectations

`eval-reid` uses the same registered ReID datasets as training. The selected dataset must expose query and gallery splits under the dataset root passed to `--data-dir`.

## Scope

`eval-reid` is available from both the CLI and Python facade via `BoxMOT.eval_reid(...)`.

## Related pages

- [Train ReID](train.md)
- [Export](export.md)
- [ReID Profiles](../config/reid.md)

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
  :command: eval_reid
    :style: table
    :prog_name: boxmot eval-reid
