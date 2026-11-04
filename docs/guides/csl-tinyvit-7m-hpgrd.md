# CSL-TinyViT-7M HP-GRD

HP-GRD is an experimental, training-only treatment for the
`csl_tinyvit_7m_v20` person-ReID model. It transfers an offline teacher's
identity relations into the RGB descriptor without changing its inference
inputs or descriptor width. This guide prepares the cache explicitly and
starts training with the bundled recipe.

The recipe is `boxmot/reid/training/configs/recipes/csl_tinyvit_7m_hpgrd.yaml`.
With 751 training identities, its training construction has 7,165,011
parameters; the matching RGB construction without privileged anatomy has
6,937,893. Those counts include training classifiers and are not final
deployment counts. Deployment removes classifiers and training-only branches
and retains the 1152-D `norm_concat_bn` descriptor.

## What is trained

| Treatment | Signal | Gradient destination | Retrieval-time cost |
| --- | --- | --- | --- |
| Fair ImageNet initialization | The same normal TinyViT initialization as the suppression arm | Existing `patch_embed` and `layers` | None |
| GlobalAP | Same-identity non-self positives and hard different-identity negatives from a stable-index memory | Existing `norm_concat_bn` path | None |
| Global graph distillation | Pairwise similarities from the frozen V20 teacher | Existing `norm_concat_bn` path | None |
| Part graph distillation | Visibility/confidence-weighted teacher part relations | Fixed mask pooling on the shared deployed feature map | None |
| Background intervention | Descriptor consistency under background replacement | Existing deployed descriptor | None |
| Semantic part dropout | Leave-one-part-out teacher relation graph | Existing deployed descriptor | None |

GlobalAP excludes the query's exact sample row, retains every other
same-identity sample as a positive, and applies top-k mining only to
different-identity negatives. HP-GRD likewise uses same-identity non-self
positives plus every different-identity negative, then balances the two
relation groups rather than letting the larger negative graph dominate.
Person identity is the sole source of pair labels. Camera IDs may diversify
sampling, but they are never provided to GlobalAP, HP-GRD, or the RGB model.
The HP-GRD gradient norm at the shared deployed feature map is capped relative
to the normal ReID objective. Intervention forwards use the same train-mode
BatchNorm domain and full batch as the primary view, then restore all running
statistics; the loss therefore cannot learn a train/eval normalization artifact
or corrupt the deployed BN descriptor.

Six-part student descriptors are fixed mask-weighted means of the shared RGB
feature map. There is no learned HP-GRD part adapter that could absorb the
teacher signal and then disappear at export: part-relation gradients reach the
existing backbone/head path directly.

HP-GRD is intentionally a composite candidate, not a one-factor loss
ablation. Relative to suppression it also uses P=24/K=4 camera-diverse
sampling, a center-loss ramp, background mosaic, and replaces the legacy
learned-anatomy objectives with fixed-mask privileged supervision. Attribute
any gain to the complete treatment unless those components are ablated
separately. Match initialization, data, seeds, and evaluation frequency
explicitly when comparing recipes.

## 1. Prepare the inputs

Use a source checkout with the training dependencies and the `yolo` extra for
metadata generation. The teacher checkpoint, dataset, and annotations are
external inputs; they are not included in the package.

1. Prepare Market1501 at `datasets/reid/Market-1501-v15.09.15`.
2. Generate `Market-1501-pav-metadata` and
   `Market-1501-mosaic-highconf-masks` using the
   [pose metadata](../modes/train.md#pose-aligned-view-mosaic) and
   [background mask](../modes/train.md#identity-preserving-background-mosaic)
   commands. The commands below reuse its `primary/` foreground masks for
   anatomical pooling and offline extraction.
3. Supply a trained `csl_tinyvit_7m_v20` checkpoint at
   `runs/reid_train/v20/best.pt`, or replace that path consistently below.
   You can train it with `--recipe csl_tinyvit_7m_v20` and the generated
   `--anatomical-metadata-dir`.

The student starts from normal ImageNet initialization (`pretrained: true`);
the V20 checkpoint is only a frozen offline teacher, not a student warm-start
or optimizer resume. Its byte-level checksum becomes part of cache provenance.
No separate human-pretraining checkpoint is required by this recipe.

## 2. Build the privileged cache

Preparation has three stages: stable dataset indexing, teacher extraction,
and cache building/validation. No automatic cache-building shell launcher is
included in this checkout. Reuse the resulting cache across seeds when the
training sample mapping, teacher, annotations, and extraction settings match.

Export the exact training mapping from the same registered dataset and root
that fine-tuning will use:

```bash
uv run --no-sync python -m boxmot.engine.commands.reid.privileged_cache index \
  --dataset market1501 \
  --data-dir datasets/reid/Market-1501-v15.09.15 \
  --output artifacts/hpgrd/market1501_v20_teacher/train-samples.json
```

The resulting index enumerates stable indices (`0..N-1`) and records
`index`, `img_path`, `pid`, and `camid`. The camera field is dataset provenance
and may guide the sampler; it does not define teacher relations or enter a
model/loss input. Do not hand-sort or rewrite the index. The teacher tensor
bundle contains:

- `global_descriptors`: `[N,Dg]`
- `part_descriptors`: `[N,P,Dp]`
- `part_visibility` and `part_confidence`: `[N,P]`
- optional `global_confidence`: `[N]`
- optional `leave_part_out_descriptors`: `[N,P,Dl]`

The part axis is semantic, not interchangeable. Cache schema v2 signs its
exact ordered names. The canonical anatomical order is `head`, `torso`,
`left_arm`, `right_arm`, `left_leg`, `right_leg`; reordering left/right or
upper/lower tensors without changing the names is invalid supervision.

Extract those tensors with the promoted V20 checkpoint and the same offline
six-part anatomical metadata used by training:

```bash
uv run --no-sync python -m boxmot.engine.commands.reid.teacher_extract \
  --teacher runs/reid_train/v20/best.pt \
  --dataset-index artifacts/hpgrd/market1501_v20_teacher/train-samples.json \
  --image-root datasets/reid/Market-1501-v15.09.15 \
  --anatomical-metadata Market-1501-pav-metadata \
  --person-mask-dir Market-1501-mosaic-highconf-masks/primary \
  --include-leave-part-out \
  --global-confidence-from-parts \
  --img-size 384 128 \
  --preprocess resize \
  --device auto \
  --batch-size 32 \
  --max-intervention-batch 32 \
  --workers 0 \
  --storage-dtype float16 \
  --output artifacts/hpgrd/market1501_v20_teacher/teacher-signals.pt
```

Precomputed pose/parser masks can be supplied with `--part-mask-input`
instead. The extractor runs the teacher only offline, aligns image/mask resize
geometry, and indexes every tensor by the stable dataset row rather than by
dataloader order. Six-part anatomical inputs default to the canonical order.
Non-canonical mask bundles must embed `part_names` or pass the ordered names
with `--part-names`; the extractor propagates them into `teacher-signals.pt`.

Build and validate the immutable cache:

```bash
uv run --no-sync python -m boxmot.engine.commands.reid.privileged_cache build \
  --tensor-input artifacts/hpgrd/market1501_v20_teacher/teacher-signals.pt \
  --dataset-index artifacts/hpgrd/market1501_v20_teacher/train-samples.json \
  --teacher-provenance runs/reid_train/v20/best.pt \
  --part-names head torso left_arm right_arm left_leg right_leg \
  --output artifacts/hpgrd/market1501_v20_teacher/privileged_graph.pt

uv run --no-sync python -m boxmot.engine.commands.reid.privileged_cache validate \
  --cache artifacts/hpgrd/market1501_v20_teacher/privileged_graph.pt \
  --dataset-index artifacts/hpgrd/market1501_v20_teacher/train-samples.json \
  --teacher-provenance runs/reid_train/v20/best.pt \
  --part-names head torso left_arm right_arm left_leg right_leg \
  --require-exact-index-file
```

Existing output files are rejected unless you explicitly pass `--overwrite`.
The builder hashes the semantic sample mapping, source files, teacher
provenance, tensor payload, and manifest. Training refuses a cache whose
dataset mapping differs from the live dataset or whose signed ordered part
names differ from the student's canonical anatomical packet.

## 3. Train the student

After cache validation succeeds, run:

```bash
boxmot train-reid \
  --recipe csl_tinyvit_7m_hpgrd \
  --data-dir datasets/reid/Market-1501-v15.09.15 \
  --anatomical-metadata-dir Market-1501-pav-metadata \
  --anatomical-person-mask-dir Market-1501-mosaic-highconf-masks/primary \
  --background-mosaic-mask-dir Market-1501-mosaic-highconf-masks \
  --hpgrd-cache-dir artifacts/hpgrd/market1501_v20_teacher/privileged_graph.pt \
  --device cuda:0 --seed 0 \
  --project runs/reid_train --name hpgrd_seed0
```

Repeat with explicit seeds and distinct run names for a multi-seed comparison.
Keep the validated offline cache unchanged. The `primary/` masks preserve the
anchor foreground; `all_people/` masks remove people from donor backgrounds.

The recipe uses camera-aware `P=24, K=4` batches, a 16,384-row GlobalAP
memory, and a warmup/hold/decay schedule over epochs 20/50/130/170. Camera
awareness only prefers diverse observations within a sampled identity; PID
alone determines positives and negatives. The memory is updated even during
warmup so it is populated when its loss activates.

`last.pt` contains the GlobalAP bank and clock for exact resume. `best.pt`
contains neither the bank nor any privileged module and remains a compact
inference artifact. The frozen V20 teacher and all generated privileged data
remain outside the deployed graph; trackers receive only the final RGB
embedding.

## Evaluation discipline

Market-1501 query/gallery is the official test split. The recipe evaluates it
only at epoch 200 to avoid repeatedly choosing treatments against the test
set. Use a separate training-identity validation split for ablation decisions,
then run the official query/gallery evaluation once for the frozen candidate.
