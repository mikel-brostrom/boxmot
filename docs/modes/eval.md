# Evaluate

`eval` measures a tracker by streaming one explicit materialized build through
the live tracker API. It never runs or downloads a detector, segmentor, or
appearance encoder.

Pass exactly one dataset selector and one build:

```bash
boxmot eval \
  --experiment mot17-ablation-yolox-lmbn \
  --build runs/builds/BUILD_ID \
  --data-root boxmot/datasets/mot \
  --tracker boosttrack

boxmot eval \
  --dataset mot17 \
  --split ablation \
  --build /srv/boxmot/builds/BUILD_ID \
  --tracker bytetrack
```

An experiment additionally fixes semantic component fingerprints. Dataset mode
uses the selected dataset adapter for ground truth. Before tracking, evaluation
verifies the build's source catalog digest, split, class taxonomy, geometry,
published requirements, and—when applicable—component fingerprints.

If a requirement is missing, evaluation reports a concrete `boxmot materialize
...` command. It does not modify the build or create a replacement.

An unbound build created before canonical experiment materialization can be
evaluated only with the explicit `--allow-noncanonical-build` escape hatch.
This relaxes the missing build-binding check; it does not turn incompatible
artifacts into compatible ones. Use it only after independently verifying the
build's source, split, detector, ReID model, geometry, and class taxonomy. For
example, to diagnose the known MMOT build on one sequence:

```bash
boxmot eval \
  --experiment mmot-obb-test-yolo11l-lmbn \
  --build faf16842d9d15fc048a50247df8ae927e7299d5760c9f461af4581cabfa279e6 \
  --data-root /Volumes/Data/MMOT \
  --tracker botsort \
  --sequence data23-1 \
  --allow-noncanonical-build
```

Omit `--sequence data23-1` to evaluate every sequence. Canonical builds remain
the default and do not need this flag.

## Sequence parallelism

Evaluation replays each sequence as one isolated spawned-process job. Every
job constructs its own tracker from the immutable tracker spec, so tracker
state and native handles never cross sequence or process boundaries.
`--n-threads` sets the maximum number of sequence worker processes. The Rich
panel reports frame progress separately for every sequence while those jobs
run.

Use `--sequence` to replay only one sequence while diagnosing a run. Repeat
the option to select more than one sequence:

```bash
boxmot eval \
  --experiment mmot-obb-test-yolo11l-lmbn \
  --build BUILD_ID \
  --sequence data23-1 \
  --tracker botsort
```

`boxmot eval` reuses the sequence frame counts already validated during dataset
setup. A standalone replay call falls back to reading only the samples table's
sequence IDs. Neither path eagerly loads masks or embeddings in the
coordinator. It marks each sequence as queued before starting the process pool.
A worker then opens only its assigned sequence and streams optional masks and
embeddings in bounded Arrow batches as frame iteration advances. Images are
decoded frame by frame when the tracker requires them.

Current materializations also use bounded Parquet row groups so workers can
prune unrelated `sample_id` ranges. Older `boxmot.dataset/v1` builds remain
readable, but builds created with large row groups can require more I/O during
the initial worker-loading phase.

## Build resolution

- An existing `--build` path is used directly.
- A build ID is looked up only below `--build-root`.
- `--build-root` defaults to `BOXMOT_BUILDS_DIR`, then the platform cache.
- There is no latest-build selection.

The selected raw data root is used to verify ground-truth provenance. Its
precedence is `--data-root`, `BOXMOT_DATASETS_DIR`, then the platform cache.

Repeated evaluations reuse hashes and image dimensions only when a source or
model file's path, device, inode, mode, size, modification time, and change
time are unchanged. This evaluation-only metadata lives in the platform cache;
the source tree is still traversed on every run so added and removed files are
detected. Materialization never uses this cache and always resolves fresh
artifact digests before establishing a build ID.

## Output

Tracker output is serialized to MOT text only at the evaluation boundary.
Reusable postprocessors can consume those files separately and never rewrite
the immutable perception build. `--compare-trackeval` is available for
supported AABB MOTChallenge datasets.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.eval
    :command: eval
    :prog_name: boxmot eval
    :depth: 0
