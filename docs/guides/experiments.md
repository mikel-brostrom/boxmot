# Experiment Workflows

Use experiment configs for the `generate`, `eval`, `tune`, and `research` modes.
An experiment selects entries from the central `boxmot/configs` catalog, so commands do
not repeat paths or numeric class IDs.

```bash
boxmot generate --experiment mot17-ablation-yolox-lmbn
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack
boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker bytetrack
```

## Detection sources

Choose the source in the experiment:

- `mot17-ablation-yolox-lmbn` runs the named detector checkpoint.
- `mot17-ablation-frcnn-lmbn` uses MOT17 public FRCNN detections.
- `mot17-ablation-precomputed` downloads the declared detections and embeddings.

The cache root is
`<project>/dets_n_embs/<dataset>/<split>/<detector-or-public-producer>/`.
Detection outputs live below `dets/`. Embeddings below `embs/` are further
partitioned by their Python or C++ producer, model format and runtime, ReID
artifact fingerprint, preprocessing policy, and crop-schema version.

Keep the same experiment, split, detection producer, ReID weights, backend,
and preprocessing overrides when later commands should reuse the same cache.

## Data and replay

Downloaded MOT-style datasets are stored under `boxmot/datasets/mot`. Most
cached replay runs do not read images; trackers that need camera-motion inputs
still load frames during replay.

Native `--tracker-backend cpp` replay can reuse the detection cache. Embedding
producer identity is the effective Python or C++ implementation that generated
the vectors, not the tracker algorithm that consumes them.
See [Embedding cache layout](../native/index.md#embedding-cache-layout).

## Outputs

- `generate` writes reusable detections and embeddings.
- `eval` writes tracker outputs, metric results, `config.source.yaml`, and `config.resolved.yaml`.
- `tune` writes trial outputs and the best parameters.
- `research` writes summaries for evaluated code proposals.

## Related pages

- [Generate](../modes/generate.md)
- [Evaluate](../modes/eval.md)
- [Evaluation and Postprocessing](evaluation.md)
- [Experiments](../config/experiments.md)
