# Experiment Workflows

An experiment binds a dataset and semantic component configuration. The same
resolved configuration contributes to the materialized build ID and is checked
again by eval, tune, and research.

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml
boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml
boxmot research --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
```

Materialization accepts only `--experiment`. The dataset, split, geometry,
detector, segmentor, ReID encoder, and class map are part of that experiment's
semantic identity. Direct component selectors cannot override an authored
`--experiment`; create another experiment when its configuration should change.

An experiment's `dataset.ref` can name a neighboring dataset YAML, such as
`kitti-2d.yaml`, or use a relative path such as `../datasets/custom.yaml`.
Local references resolve from the experiment's directory. A neighboring file
takes precedence over a built-in dataset with the same filename. Bare catalog
IDs and filenames still select built-in datasets when no local config exists.

Evaluation and tuning also accept `--dataset` and `--detector` with an optional `--reid`
selector as shorthand for an authored catalog experiment:

```bash
boxmot eval --dataset mot17 --split ablation \
  --detector yolox-x-mot17 --reid lmbn-n-duke --tracker botsort
```

This selects `mot17/ablation-yolox-lmbn.yaml` and reuses the exact same
materialization and build as `--experiment mot17/ablation-yolox-lmbn.yaml`.
The authored experiment supplies the detector checkpoint, class map, and other
semantic settings. No match or multiple matches produce an error; use
`--experiment` to disambiguate or author a configuration absent from the catalog.
Append `/CHECKPOINT` to the detector profile when needed to narrow a match.
Omitting `--reid` matches only experiments without a ReID profile.

Evaluation and tuning prepare or reuse a canonical build when `--build` is
omitted. Their dataset-only forms without a detector require an explicit
`--build`. Automatic preparation remains experiment-backed, whether selected
by filename or component shorthand; source and semantic fingerprints must
match for reuse. Research requires an authored experiment and an explicit
build.

The build manifest records resolved artifact hashes, source and taxonomy
digests, stage fingerprints, publish flags, shard counts/hashes, and
completeness. Executor worker count, retry count, prefetch depth, and shard size
do not change semantic fingerprints.
