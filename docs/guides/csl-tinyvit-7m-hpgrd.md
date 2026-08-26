# CSL-TinyViT-7M HP-GRD

HP-GRD is an experimental, training-only treatment for the promoted
`csl_tinyvit_7m_v20` person-ReID model. It targets the gap between the current
89.96% Market-1501 mAP run and 93% mAP without enlarging or changing the
deployed model. The 93% value is a research target, not a measured result.

The recipe is `boxmot/reid/training/configs/recipes/csl_tinyvit_7m_hpgrd.yaml`.
Its training graph has 7,165,011 parameters; the pruned deployed RGB graph has
6,937,893 and still emits `norm_concat_bn`. Every pose, parsing, teacher,
memory, and intervention object is removed from `best.pt`.

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
separately. The launcher does keep initialization and official-query
evaluation frequency matched between the two requested arms.

## 1. Reuse V20 as the offline teacher

The self-contained treatment uses the existing promoted checkpoint:

```text
runs/csl_tinyvit_7m_fix/a0_model_fixes_applied/best.pt
```

The immutable file has one training-only role: it is the frozen offline
teacher used to generate global, semantic-part, and leave-one-part-out relation
targets.

The HP-GRD student does not load V20 weights. It uses `pretrained: true`, the
same normal ImageNet TinyViT initialization as the suppression arm, with a
fresh ReID head and identity classifiers. This makes the comparison fair:
neither arm receives the promoted model as a student warm-start. HP-GRD is not
an optimizer resume or a continuation of the complete 89.96% run; V20 supplies
only frozen relational targets generated before training.

No separate human-pretraining run and no larger teacher checkpoint are
required. This version is V20 graph self-distillation with training-only pose
and segmentation guidance. The cache manifest records the V20 checkpoint's
byte-level SHA-256, so replacing it at the same path makes an existing cache
fail provenance validation.

## 2. Automatic privileged-cache preparation

The ablation launcher builds the HP-GRD inputs automatically when its cache is
missing. It performs three deterministic steps before training:

1. Export the exact stable Market-1501 training mapping.
2. Run the frozen V20 checkpoint over the clean images, six-part masked-in
   views, and six leave-one-part-out views derived from the existing PAV
   metadata and person masks.
3. Build and validate the immutable privileged graph cache against both the
   dataset mapping and V20 teacher checksum.

The first extraction is computationally expensive because the teacher must
process every training crop and its semantic interventions. The generated
artifacts are stored under `artifacts/hpgrd/market1501_v20_teacher` and reused
by later seeds; selecting another seed does not rerun the teacher. A present
but invalid cache is rejected rather than silently overwritten.

`AUTO_BUILD_HPGRD_CACHE=1` enables this preparation and is the launcher
default. `HPGRD_EXTRACT_DEVICE`, `HPGRD_EXTRACT_BATCH_SIZE`, and
`HPGRD_EXTRACT_WORKERS` tune offline extraction independently of training.
`HPGRD_TEACHER_WEIGHTS` overrides the default V20 teacher path when needed.
`HPGRD_ARTIFACT_DIR` can move all three generated artifacts together.

To prepare and validate the launcher-compatible cache without starting a
training run:

```bash
PREPARE_ONLY=1 ARMS=hpgrd ./ablation_csl_tinyvit_7m_new_variants.sh
```

The launcher signs the full teacher, dataset, anatomical-input, extractor
configuration, and teacher-signal hashes into sidecar/cache provenance. The
following low-level commands illustrate the underlying pipeline, but the raw
build command intentionally omits that launcher-specific signature; use
`PREPARE_ONLY=1` for an artifact the ablation launcher will accept.

Export the exact training mapping from the same registered dataset and root
that fine-tuning will use:

```bash
uv run python -m boxmot.engine.reid.privileged_cache index \
  --dataset market1501 \
  --data-dir boxmot/datasets/reid/Market-1501-v15.09.15 \
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
uv run python -m boxmot.engine.reid.teacher_extractor \
  --teacher runs/csl_tinyvit_7m_fix/a0_model_fixes_applied/best.pt \
  --dataset-index artifacts/hpgrd/market1501_v20_teacher/train-samples.json \
  --image-root boxmot/datasets/reid/Market-1501-v15.09.15 \
  --anatomical-metadata Market-1501-pav-metadata-clean \
  --person-mask-dir Market-1501-mosaic-highconf-person-masks \
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
uv run python -m boxmot.engine.reid.privileged_cache build \
  --tensor-input artifacts/hpgrd/market1501_v20_teacher/teacher-signals.pt \
  --dataset-index artifacts/hpgrd/market1501_v20_teacher/train-samples.json \
  --teacher-provenance runs/csl_tinyvit_7m_fix/a0_model_fixes_applied/best.pt \
  --part-names head torso left_arm right_arm left_leg right_leg \
  --output artifacts/hpgrd/market1501_v20_teacher/privileged_graph.pt

uv run python -m boxmot.engine.reid.privileged_cache validate \
  --cache artifacts/hpgrd/market1501_v20_teacher/privileged_graph.pt \
  --dataset-index artifacts/hpgrd/market1501_v20_teacher/train-samples.json \
  --teacher-provenance runs/csl_tinyvit_7m_fix/a0_model_fixes_applied/best.pt \
  --part-names head torso left_arm right_arm left_leg right_leg \
  --require-exact-index-file
```

The builder hashes the semantic sample mapping, source files, teacher
provenance, tensor payload, and manifest. Training refuses a cache whose
dataset mapping differs from the live dataset or whose signed ordered part
names differ from the student's canonical anatomical packet.

## 3. Run the requested ablation

The self-contained launcher starts with these existing inputs:

- `runs/csl_tinyvit_7m_fix/a0_model_fixes_applied/best.pt`
- `boxmot/datasets/reid/Market-1501-v15.09.15`
- `Market-1501-pav-metadata-clean`
- `Market-1501-mosaic-highconf-masks`
- `Market-1501-mosaic-highconf-person-masks`

The privileged cache is an output of launcher preparation, not a checkpoint
the user must obtain. The first mask root supplies the background compositor's
`primary/` and `all_people/` trees; the second supplies per-crop foreground
masks for fixed anatomical pooling and offline teacher extraction.

Run only multilevel suppression and HP-GRD from the repository root:

```bash
ARMS=suppression,hpgrd SEEDS=0 DEVICE=0 \
  ./ablation_csl_tinyvit_7m_new_variants.sh
```

For a multi-seed comparison, set `SEEDS=0,1,2`. Runs are sequential, and the
same validated offline cache is shared across seeds.

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
