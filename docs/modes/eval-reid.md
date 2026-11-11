# Evaluate ReID

Use `eval-reid` to score one trained ReID checkpoint on one dataset's query/gallery split.

## Examples

Use a `best.pt` produced by [training](train.md) and replace `/data/reid` with
the parent of your local datasets. Both tabs use the same evaluation workflow
and save the same JSON reports. Pass Python options as keyword arguments;
omitted optional values use the workflow defaults.

!!! example

    === "CLI"

        Evaluate a checkpoint on Market1501:

        ```bash
        boxmot eval-reid \
          --weights runs/reid_train/osnet_market/best.pt \
          --dataset market1501 \
          --data-dir /data/reid \
          --device cuda:0
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

    === "Python"

        Evaluate a checkpoint on Market1501:

        ```python
        from boxmot.engine.commands.reid.evaluate import main as evaluate_reid

        if __name__ == "__main__":
            result = evaluate_reid(
                weights="runs/reid_train/osnet_market/best.pt",
                dataset="market1501",
                data_dir="/data/reid",
                device="cuda:0",
            )
            print(result)
        ```

        Save results to a custom directory:

        ```python
        from boxmot.engine.commands.reid.evaluate import main as evaluate_reid

        if __name__ == "__main__":
            result = evaluate_reid(
                weights="runs/reid_train/lmbn_joint/best.pt",
                dataset="msmt17",
                data_dir="/data/reid",
                batch_size=128,
                output="runs/reid_eval",
            )
            print(result)
        ```

## What it does

`eval-reid` loads a trained checkpoint, rebuilds the matching backbone, extracts query and gallery embeddings, computes the distance matrix, and reports ranking metrics.

If the checkpoint stores the model architecture, you can omit `--model`. Otherwise pass the architecture explicitly.

Use a PyTorch `.pt` checkpoint. This command rebuilds the PyTorch model; it does
not benchmark exported ONNX, TensorRT, Core ML, or TFLite artifacts. Device
values are PyTorch names: `cpu` (default), `mps`, or `cuda:0`. A bare `0` is not
accepted by this workflow.

Keep `hparams.json` beside the checkpoint. Image size comes from `--imgsz`, then
that file, then the `256,128` fallback. Preprocessing comes from `--preprocess`,
the checkpoint, then `hparams.json`; the final fallback is `resize`. Flip TTA
uses the saved hparams unless explicitly overridden. For a moved checkpoint
without hparams, supply its training crop explicitly, for example
`--imgsz 384,128 --preprocess resize --no-flip-tta`.

## Main outputs

The command reports and saves:

- `mAP`
- `rank1`
- `rank5`
- `rank10`
- measured inference latency, including milliseconds per batch and per image

Latency measurement uses five warmup passes and 30 timed passes by default. Set
`--latency-iters 0` to disable it. Timing covers model forwards on the first
query batch, including flip TTA when enabled. It excludes image decoding,
preprocessing, host-to-device transfer, and query/gallery distance computation.
JSON metric values are fractions in `[0, 1]`; console metrics are percentages.

By default the JSON summary is written next to the checkpoint as
`eval_<dataset>_<inference-feature>.json` when an inference feature is active,
or as `eval_<dataset>.json` otherwise. With `--output`, the filename is
`eval_<model>_<dataset>[_<inference-feature>].json`.

## Dataset expectations

`eval-reid` uses the same [registered ReID datasets](train.md#supported-datasets)
as training. Supply the complete dataset layout, including its training split:
the reader constructs all three splits, even though only query/gallery crops
are scored. The command does not download missing data.

Ordinary embeddings are L2-normalized and compared with cosine distance.
Ranking removes gallery entries sharing both identity and camera with a query;
queries without a remaining positive are excluded. Structured CSL-TinyViT
features use their specialized distance functions. Use the same crop,
descriptor, and flip-TTA settings when comparing checkpoints.

## Scope

The Python examples call the engine workflow, including JSON persistence. For
an in-memory result without writing a report, use `ReIDEvaluationConfig` and
`evaluate_reid` from `boxmot.reid.evaluation.runner`. Dataset and metric code
remain available from the `boxmot.reid` domain subpackages.

## Related pages

- [Train ReID](train.md)
- [Compare ReID](compare-reid.md)
- [Export](export.md)
- [ReID Profiles](../config/reid.md)

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.reid.evaluate
    :command: eval_reid
    :depth: 0
    :style: table
    :prog_name: boxmot eval-reid
