# Train ReID

Use `train` to fit a ReID backbone on a supported person or vehicle re-identification dataset.

## Examples

!!! example

    === "CLI"

        Train on Market1501:

        ```bash
        boxmot train \
          --model osnet_x0_25 \
          --dataset market1501 \
          --data-dir /data/reid \
          --device 0
        ```

        Joint training on multiple datasets:

        ```bash
        boxmot train \
          --model lmbn_n \
          --dataset market1501,duke,cuhk03 \
          --data-dir /data/reid \
          --loss triplet \
          --preprocess crop_letterbox \
          --epochs 120 \
          --project runs/reid_train \
          --name lmbn_joint
        ```

## Core idea

`train` builds a ReID backbone, loads one or more registered ReID datasets, and optimizes the model with either softmax or triplet-style training.

The crop preprocessing you choose here should match the preprocessing used later at inference time.

## Supported datasets

The built-in dataset registry currently includes common ReID benchmarks such as:

- `market1501`
- `duke` / `dukemtmcreid`
- `cuhk03`
- `msmt17`
- `msmt17_merged`

You pass the dataset root through `--data-dir`, and BoxMOT resolves the expected subdirectory layout for the selected dataset.

## Main outputs

Training writes an experiment directory under `--project/--name`, typically containing:

- best and last checkpoints
- training logs and metrics
- periodic validation results

When training finishes, BoxMOT reports the best checkpoint path along with the best validation `mAP` and `rank-1` score.

## Resuming and evaluation during training

- Use `--resume` with a checkpoint directory or `last.pt` file to continue an interrupted run.
- Use `--eval-interval` to control how often validation runs during training.
- Use `--eval-datasets` for extra cross-domain checks during training.

## Scope

`train` is currently a CLI workflow. The documented high-level `Boxmot` Python facade does not expose a matching `train(...)` method.

## Related pages

- [Evaluate ReID](eval-reid.md)
- [Export](export.md)
- [ReID Profiles](../config/reid.md)

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: train
    :style: table
    :prog_name: boxmot train