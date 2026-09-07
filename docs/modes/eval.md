# Evaluate

`eval` measures a tracker by streaming an immutable materialized build through
the live tracker API. With `--experiment`, `--build` is optional: when omitted,
BoxMOT first materializes (or reuses) a canonical build compatible with the
selected tracker and then evaluates it. Detector, segmentor, and
appearance-encoder inference happens only during that preparation step, never
during replay. Automatic preparation publishes image references, embeddings
for appearance-capable trackers, and masks when the selected tracker requires
them. Motion-only trackers such as SFSORT skip the embedding stage entirely.

Preparation caches detector output separately from ReID embeddings. If a
compatible dataset, detector, geometry, and class mapping have already been
materialized, changing the ReID model skips detector inference and runs the
remaining derived stages. The cache lives below
`<selected-build-root>/.cache/detect`; with the default root, BoxMOT can also
import an identical v1 build or seed detections from another compatible v1
build in the former platform-cache location.
Detector-native masks or embeddings are not represented by this geometry cache,
so builds that require either output run their detector stage normally.

Pass exactly one experiment or dataset selector. Experiment mode can prepare
its own build:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --data-root datasets/mot \
  --device mps \
  --tracker boosttrack
```

`--device` selects the detector, segmentor, and ReID execution device for this
automatic preparation. It is rejected with an explicit `--build`, where no
perception model runs.

After materialization completes, evaluation consumes the exact path returned
by that build operation. It does not scan `--build-root`, select a latest
directory, or risk replaying another experiment's build.

Pass `--build` to reuse a specific build. Dataset-only mode always requires it
because a dataset config does not select the detector and ReID components
needed for materialization:

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --build /srv/boxmot/materializations/BUILD_ID \
  --tracker bytetrack
```

An experiment additionally fixes semantic component fingerprints. Dataset mode
uses the selected dataset adapter for ground truth. Before tracking, evaluation
verifies the build's source catalog digest, split, class taxonomy, geometry,
published requirements, and—when applicable—component fingerprints.

If an explicitly selected build is missing or incompatible, evaluation fails
without modifying it or creating a replacement.

An unbound build created before canonical experiment materialization can be
evaluated only with the explicit `--allow-noncanonical-build` escape hatch.
This relaxes the missing build-binding check; it does not turn incompatible
artifacts into compatible ones. Use it only after independently verifying the
build's source, split, detector, ReID model, geometry, and class taxonomy. For
example, to diagnose the known MMOT build on one sequence:

```bash
boxmot eval \
  --experiment mmot-obb/test-yolo11l-lmbn.yaml \
  --build faf16842d9d15fc048a50247df8ae927e7299d5760c9f461af4581cabfa279e6 \
  --data-root datasets/mot \
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
  --experiment mmot-obb/test-yolo11l-lmbn.yaml \
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
decoded only when the resolved tracker requires source pixels. A
dimensions-only tracker such as SFSORT receives width and height from cached
sample metadata without opening the source image.

Current materializations also use bounded Parquet row groups so workers can
prune unrelated `sample_id` ranges. Older `boxmot.dataset/v1` builds remain
readable, but builds created with large row groups can require more I/O during
the initial worker-loading phase.

## Build resolution

- When `--build` is omitted with `--experiment`, BoxMOT materializes the
  deterministic build below `--build-root`; an identical complete build is
  validated and reused.
- An existing `--build` path is used directly.
- A build ID is looked up only below `--build-root`.
- `--build-root` defaults to `BOXMOT_BUILDS_DIR`, then
  `./runs/materializations`.
- Dataset-only evaluation requires `--build`.
- There is no latest-build selection.

The selected raw data root is used to verify ground-truth provenance. It
defaults to `./datasets/mot`; pass `--data-root` explicitly to use another
location. When the selected split is absent, its configured Hugging Face
`per_split` resource is downloaded before materialization or ground-truth
validation. Existing populated splits are reused without downloading;
archive-backed datasets still require explicit setup.

Repeated evaluations reuse hashes and image dimensions only when a source or
model file's path, device, inode, mode, size, modification time, and change
time are unchanged. This shared metadata lives in the platform cache;
the source tree is still traversed on every run so added and removed files are
detected. This metadata cache is separate from the content-addressed detector
output cache; materialization still establishes the source and model content
digests before selecting either a complete build or reusable detections.

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
