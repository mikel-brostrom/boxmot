# Compare ReID

Use `compare-reid` to evaluate several ReID checkpoints across several target
datasets and collect the results in one cross-domain comparison.

## Example

```bash
boxmot compare-reid \
  --weights runs/reid_train/market/best.pt \
  --weights runs/reid_train/msmt17/best.pt \
  --label market-model \
  --label msmt17-model \
  --target market1501=/data/reid \
  --target msmt17=/data/reid \
  --device 0 \
  --output runs/reid_cross_domain
```

Repeat `--weights` for every checkpoint and `--target DATASET=DATA_DIR` for
every evaluation target. If you pass `--label`, provide one label per
checkpoint. An explicit `--model` may be supplied once for all checkpoints or
once per checkpoint when the architecture cannot be recovered from metadata.

## Cross-domain filtering

By default, checkpoint metadata is used to skip a target that matches that
checkpoint's training dataset. Pass `--include-same-dataset` to evaluate those
pairs too. If a checkpoint does not identify its training dataset, none of its
targets are skipped.

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

## Related pages

- [Evaluate ReID](eval-reid.md)
- [Train ReID](train.md)
- [Export](export.md)

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: compare_reid
    :style: table
    :prog_name: boxmot compare-reid
