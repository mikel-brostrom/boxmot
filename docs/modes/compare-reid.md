# Compare ReID

Use `compare-reid` to evaluate several ReID checkpoints across several target
datasets and collect the results in one cross-domain comparison.

## Example

Use checkpoints from two training runs and replace `/data/reid` with the
parent of your local datasets. Both tabs run the same comparison and save its
reports. Python uses tuples for repeated CLI options.

=== "CLI"

    ```bash
    boxmot compare-reid \
      --weights runs/reid_train/market/best.pt \
      --weights runs/reid_train/msmt17/best.pt \
      --label market-model \
      --label msmt17-model \
      --target market1501=/data/reid \
      --target msmt17=/data/reid \
      --device cuda:0 \
      --output runs/reid_cross_domain
    ```

=== "Python"

    ```python
    from boxmot.engine.commands.reid.compare import main as compare_reid

    if __name__ == "__main__":
        result = compare_reid(
            weights=(
                "runs/reid_train/market/best.pt",
                "runs/reid_train/msmt17/best.pt",
            ),
            label=("market-model", "msmt17-model"),
            target=("market1501=/data/reid", "msmt17=/data/reid"),
            device="cuda:0",
            output="runs/reid_cross_domain",
        )
        print(result["summary"])
    ```

Repeat `--weights` for every checkpoint and `--target DATASET=DATA_DIR` for
every evaluation target. If you pass `--label`, provide one label per
checkpoint. An explicit `--model` may be supplied once for all checkpoints or
once per checkpoint when the architecture cannot be recovered from metadata.

Supply PyTorch `.pt` checkpoints and complete local ReID dataset layouts.
Device values are `cpu`, `mps`, or `cuda:0`; this command uses the same
[preprocessing and timing rules](eval-reid.md#what-it-does) as `eval-reid`.
Retain each run's `hparams.json` beside its checkpoint so crop size, flip TTA,
and training-dataset provenance remain available.

## Cross-domain filtering

By default, `hparams.json` (or checkpoint metadata when unavailable) is used to
skip targets matching any of that checkpoint's training datasets. Pass
`--include-same-dataset` to evaluate those
pairs too. If a checkpoint does not identify its training dataset, none of its
targets are skipped and their cross-domain status is unknown. Filtering is
name-based; it does not detect overlapping images or identities. In particular,
a model trained on `msmt17_merged` has seen MSMT17 query/gallery images, even
though the merged and unmerged dataset names differ.

Use `--continue-on-error` to record a failed checkpoint/target pair and continue
the matrix. The default is to stop at the first error.

## Outputs

The default output directory is `runs/reid_cross_domain`. It contains:

- `cross_domain_results.json`, with aggregate counts and all result rows
- `cross_domain_results.md`, with a readable comparison table
- one subdirectory per model label, containing the individual eval JSON files
- `map_vs_latency.png` when latency results are available

The latency plot is enabled by the default timed passes. Set
`--latency-iters 0` to disable latency measurement and the plot.
The plot also requires Matplotlib. Aggregate JSON and Markdown use fractional
metrics (`0.90` means 90% mAP). The JSON records successful, skipped, and failed
pairs; per-model evaluation files are written only for successful pairs.

## Related pages

- [Evaluate ReID](eval-reid.md)
- [Train ReID](train.md)
- [Export](export.md)

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.reid.compare
    :command: compare_reid
    :depth: 0
    :style: table
    :prog_name: boxmot compare-reid
