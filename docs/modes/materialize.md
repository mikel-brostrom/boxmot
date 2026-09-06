# Materialize

Use `materialize` to publish an immutable, keyed perception dataset for
evaluation, tuning, or research. A successful build contains Arrow-written,
zstd-compressed Parquet shards plus a `boxmot.dataset/v1` manifest and success
marker. Incomplete staging directories are resumable but are never visible as
builds.

## Select an experiment

`--experiment YAML_FILE` is the only materialization input. The experiment
owns the dataset, split, geometry, detector checkpoint, optional segmentor and
ReID encoder, and evaluation class map. Materialization does not accept
command-line replacements for those semantic values.

Use a built-in experiment's catalog-relative YAML filename or pass an explicit
experiment YAML path. To change a dataset, split, or perception component,
create a different experiment that references the appropriate reusable config
assets. This guarantees that every published build carries the dataset and
experiment identity expected by eval, tune, and research.

```bash
# Catalog-backed experiment
boxmot materialize \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --data-root datasets/mot \
  --build-root runs/materializations \
  --device mps

# Standard MMOT 8-band test split
boxmot materialize \
  --experiment mmot-obb/test-yolo11l-lmbn.yaml \
  --data-root datasets/mot \
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

ReID crop extraction is automatic: AABB detections use clipped axis-aligned
crops, while OBB detections use the canonical rectified OBB transform. The crop
is not an experiment override. Built-in ReID profiles do not consume detection
masks; a custom mask-dependent encoder declares that requirement through its
encoder contract.

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

`--build-root` overrides `BOXMOT_BUILDS_DIR`; otherwise BoxMOT uses
`./runs/materializations`. This name distinguishes immutable perception datasets
from C++ compilation output. A build ID resolves only below that root. An
already complete identical build is validated and reused without mutation.

Detector output is cached independently below
`<selected-build-root>/.cache/detect` (by default,
`./runs/materializations/.cache/detect`). Its content identity includes the
source dataset fingerprint, detector artifact and inference settings, geometry,
and class mapping, but not the experiment's ReID encoder. Consequently, a new
build that changes only ReID skips detector inference and runs its remaining
derived stages, normally embedding and finalization. A compatible published v1
build can seed a missing cache automatically. With the repository-local default,
the former platform-cache build root is also searched: an identical complete
build is imported atomically, while another compatible build can seed detector
output only. Imported rows are re-keyed to the new build ID, so every published
build keeps canonical `build:sample:detection` instance IDs.

The shared cache contains canonical sample metadata and detector instances.
Builds that require detector-native masks or detector-native embeddings bypass
it; masks from an explicit segmentor and embeddings from an explicit ReID
encoder remain normal derived stages and can reuse it.

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
created only when its stage has pending work. A detection-cache hit therefore
does not initialize or load the detector. A runtime is retained across that
stage's retry attempts and released when the stage finishes before the next
accelerator model is loaded.

In an interactive terminal, BoxMOT keeps setup, model loading, build-lock
status, and all materialization stages in one Rich progress panel. Percentages
advance only after a Parquet shard is durably checkpointed; the panel also
shows resumed shard counts, elapsed time, rate, ETA, retries, and the final
build ID. When stderr is redirected or no terminal is available, the same
events are emitted as ordinary log lines.

Raw datasets default to `./datasets/mot`; pass `--data-root` explicitly to use
another location. Downloaded tracking datasets stay outside the importable
`boxmot.datasets` Python package. If the experiment's selected split is absent,
BoxMOT downloads its configured Hugging Face `per_split` resource before
cataloging; an existing populated split is reused unchanged. Archive-backed
datasets still require explicit setup.

## Consume the build

`tune` and `research` require an explicit build ID or path. `eval` accepts the
same explicit form, but can also run canonical materialization automatically
when an experiment is selected and `--build` is omitted:

When the build root is configured, the compact form is `--build BUILD_ID`.
The checkout-oriented examples below instead use the complete build path so
there is no implicit root lookup.

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml \
  --build runs/materializations/BUILD_ID --data-root datasets/mot
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml \
  --data-root datasets/mot
boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml \
  --build runs/materializations/BUILD_ID --data-root datasets/mot
boxmot research --experiment mot17/ablation-yolox-lmbn.yaml \
  --build runs/materializations/BUILD_ID --data-root datasets/mot
```

Automatic eval preparation resolves the deterministic build for the selected
experiment; it never selects a “latest” build. Tune and research do not
materialize implicitly. Legacy NumPy, NPZ, and text-only cache roots are
unsupported and are never migrated or deleted.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.materialize
    :command: materialize
    :prog_name: boxmot materialize
    :depth: 0
