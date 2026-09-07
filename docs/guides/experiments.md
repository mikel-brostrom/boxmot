# Experiment Workflows

An experiment binds a dataset and semantic component configuration. The same
resolved configuration contributes to the materialized build ID and is checked
again by eval, tune, and research.

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml
boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
boxmot research --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
```

Materialization accepts only `--experiment`. The dataset, split, geometry,
detector, segmentor, ReID encoder, and class map are part of that experiment's
semantic identity and have no command-line overrides. Create another
experiment when one of them should change.

Evaluation can use either an experiment or dataset selector. Tune and research
require an experiment. Materialization is always experiment-backed. Evaluation
materializes or reuses a compatible canonical build when `--build` is omitted;
dataset-only evaluation, tune, and research still require an explicit build.

The build manifest records resolved artifact hashes, source and taxonomy
digests, stage fingerprints, publish flags, shard counts/hashes, and
completeness. Executor worker count, retry count, prefetch depth, and shard size
do not change semantic fingerprints.
