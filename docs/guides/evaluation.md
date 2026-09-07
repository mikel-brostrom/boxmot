# Evaluation and Postprocessing

Evaluation has two independent inputs:

1. A dataset adapter owns raw ground truth and its source-catalog digest.
2. A complete `boxmot.dataset/v1` build owns keyed perception artifacts.

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --build runs/materializations/BUILD_ID \
  --data-root datasets/mot \
  --tracker boosttrack
```

The evaluator validates that those inputs describe the same source, split,
taxonomy, and geometry before replay. Experiment mode also validates semantic
component fingerprints. Omit `--build` to materialize or reuse a compatible
canonical build before replay:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --data-root datasets/mot \
  --tracker boosttrack
```

Supplying `--build` skips automatic preparation and evaluates exactly that
build.

Automatic preparation stores detector results independently from ReID output
under `<selected-build-root>/.cache/detect` (by default,
`runs/materializations/.cache/detect`). Experiments that share the source
dataset, detector, geometry, and class mapping therefore skip repeated detector
inference and run only their remaining derived stages. Detector-native masks or
embeddings bypass this geometry-only cache.

Tracker requirements are checked against published artifacts. For example, a
configuration with `use_embeddings: true` requires embeddings in the build;
Sam2Mot requires full-frame detection-aligned masks and frames. Missing inputs
produce an actionable materialization error.

Cached replay is sequence-parallel: each sequence is handled by a spawned
process with its own tracker instance. Use `--n-threads` to cap the active
sequence processes; the evaluation UI shows a separate frame-progress row for
each sequence. Use `--sequence NAME` to limit a diagnostic run to one sequence;
repeat the option to select several. The parent reuses the frame counts already
validated during dataset setup and does not reopen perception payloads before
launch. Each worker reads detections for its assigned sequence and defers keyed
masks, embeddings, and image decoding until frame iteration, so optional
payloads are not replicated eagerly across the process pool.

Postprocessing such as interpolation applies to serialized tracker results,
not to the immutable build. Ground truth remains under the selected dataset
adapter rather than being embedded into the generic perception dataset.

Legacy `.npy`, `.npz`, and text-only perception caches are unsupported. They
are not read, migrated, or deleted.
