# Materialize

Use `materialize` to publish an immutable, keyed perception dataset for
evaluation, tuning, or research. A successful build contains Arrow-written,
zstd-compressed Parquet shards plus a `boxmot.dataset/v1` manifest and success
marker. Incomplete staging directories are resumable but are never visible as
builds.

## Select an experiment

`--experiment ID_OR_YAML` is the only materialization input. The experiment
owns the dataset, split, geometry, detector checkpoint, optional segmentor and
ReID encoder, and evaluation class map. Materialization does not accept
command-line replacements for those semantic values.

Use a built-in experiment ID or pass an explicit experiment YAML. To change a
dataset, split, or perception component, create a different experiment that
references the appropriate reusable config assets. This guarantees that every
published build carries the dataset and experiment identity expected by eval,
tune, and research.

```bash
# Catalog-backed experiment
boxmot materialize \
  --experiment mot17-ablation-yolox-lmbn \
  --data-root boxmot/datasets/mot \
  --build-root runs/builds \
  --device mps

# Standard MMOT 8-band test split
boxmot materialize \
  --experiment mmot-obb-test-yolo11l-lmbn \
  --data-root /Volumes/Data/MMOT \
  --device mps
```

NumPy frames selected by an experiment's dataset must be `uint8` HWC arrays
with at least three channels.
Eight-channel MMOT frames reproduce the 3-channel OBB checkpoint's training
transform: source indices `[1, 2, 4]` enter the OpenCV/Ultralytics BGR boundary,
then the standard BGR-to-RGB conversion presents `[4, 2, 1]` to the network
(one-based bands 5, 3, and 2). Other arrays wider than three channels use their
first three. These dataset images are not the unsupported legacy positional
`.npy` result caches.

The built-in MMOT OBB experiments also set
`reid.crop_strategy: perspective`. Each detected oriented rectangle is
perspective-rectified with its authored width/height orientation before LMBN
embedding inference, matching the published benchmark cache. Using its
enclosing AABB or the canonical affine transform instead produces a different
appearance cache and therefore a different immutable build ID.

Image references are published by default, masks are disabled by default, and
embeddings are enabled by default. Model artifacts are resolved and hashed
before the build ID is established; materialization never changes that semantic
identity after work begins.

Use `--device cpu`, `--device mps`, or a CUDA selector such as
`--device cuda:0` (`--device 0` is equivalent) to override the execution device
for every perception component in the build. If omitted, each resolved
component keeps its configured device; a configured `auto` resolves to the
command default (`cpu`). Unavailable explicit accelerators fail before model
loading with an actionable error. The effective per-component devices are shown
in the Rich panel and included in the immutable build fingerprint.

## Build location and resume

`--build-root` overrides `BOXMOT_BUILDS_DIR`; otherwise BoxMOT uses the platform
cache directory for `boxmot/builds`. A build ID resolves only below that root.
An already complete identical build is validated and reused without mutation.

Use `--plan executor.yaml` for local executor settings and repeat
`--set stage.field=value` for typed YAML overrides. Unknown fields fail. Resume
is enabled by default; use `--no-resume` to refuse stale or interrupted staging
state.

The first catalog pass hashes each source file and stores its dimensions and
digest in the platform cache. Later invocations reuse that metadata only when
the canonical path, device, inode, mode, size, modification time, and change
time all still match. Pending stage batches consult the same cache before
decoding, while completed stages do not reopen source frames. Any identity
change forces a fresh content hash and a mismatched digest stops publication.
Materialization and evaluation share this cache; unchanged entries from the
older evaluation-only cache are imported automatically without rehashing.

Perception runtimes are also lazy: a detector, segmentor, or ReID encoder is
created only when its stage has pending shards. The runtime is retained across
that stage's retry attempts and released when the stage finishes before the
next accelerator model is loaded.

In an interactive terminal, BoxMOT keeps setup, model loading, build-lock
status, and all materialization stages in one Rich progress panel. Percentages
advance only after a Parquet shard is durably checkpointed; the panel also
shows resumed shard counts, elapsed time, rate, ETA, retries, and the final
build ID. When stderr is redirected or no terminal is available, the same
events are emitted as ordinary log lines.

Raw dataset roots resolve in this order: `--data-root`,
`BOXMOT_DATASETS_DIR`, then the platform cache directory for
`boxmot/datasets`. Repository-local datasets are not discovered implicitly.

## Consume the build

Downstream commands always require an explicit build ID or path:

When the build root is configured, the compact form is `--build BUILD_ID`.
The checkout-oriented examples below instead use the complete build path so
there is no implicit root lookup.

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn \
  --build runs/builds/BUILD_ID --data-root boxmot/datasets/mot
boxmot tune --experiment mot17-ablation-yolox-lmbn \
  --build runs/builds/BUILD_ID --data-root boxmot/datasets/mot
boxmot research --experiment mot17-ablation-yolox-lmbn \
  --build runs/builds/BUILD_ID --data-root boxmot/datasets/mot
```

There is no implicit materialization and no “latest build” selection. Legacy
NumPy, NPZ, and text-only cache roots are unsupported and are never migrated or
deleted.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.materialize
    :command: materialize
    :prog_name: boxmot materialize
    :depth: 0
