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

        Joint training from ReID data YAMLs:

        ```bash
        boxmot train \
          --model csl_tinyvit_23m \
          --data market1501.yaml \
          --data duke.yaml \
          --epochs 120 \
          --device 0
        ```

        Train from a BoxMOT ReID config:

        ```bash
        boxmot train --cfg custom_config.yaml
        ```

        Explicit CLI flags override the config:

        ```bash
        boxmot train --cfg custom_config.yaml --epochs 3
        ```

        Example `market1501.yaml`:

        ```yaml
        dataset: market1501
        path: ../datasets/Market-1501-v15.09.15
        train: bounding_box_train
        query: query
        gallery: bounding_box_test
        download: |
          from pathlib import Path
          Path(yaml["path"]).mkdir(parents=True, exist_ok=True)
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

Alternatively, pass one or more `--data` YAML configs. YAML `path` values are resolved relative to the YAML file, and `download` is a local Python block executed only when that root is missing or empty. Built-in ReID datasets still use their registered parsers; `train`, `query`, and `gallery` are saved in hparams as dataset metadata.

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

`train` is available from both the CLI and the high-level `BoxMOT.train(...)` Python facade.

```python
from boxmot import BoxMOT

model = BoxMOT("mobilenetv4")
model.train(cfg="mobilenetv4_custom.yaml")
```

When the first positional argument matches a registered ReID training recipe or backbone, it is used as the training profile; detector names still configure tracking detectors. A ReID weight filename can also seed the training profile while binding the object to that weight for later export or embedding:

```python
reid = BoxMOT(reid="mobilenetv4.pt")
reid.train(cfg="custom_config.yaml")
```

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
