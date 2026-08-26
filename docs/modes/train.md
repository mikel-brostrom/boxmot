# Train ReID

Use `train-reid` to fit a ReID backbone on a supported person or vehicle re-identification dataset.

## Examples

!!! example

    === "CLI"

        Train on Market1501:

        ```bash
        boxmot train-reid \
          --model osnet_x0_25 \
          --dataset market1501 \
          --data-dir /data/reid \
          --device 0
        ```

        Joint training on multiple datasets:

        ```bash
        boxmot train-reid \
          --model lmbn_n \
          --dataset market1501,duke,cuhk03 \
          --data-dir /data/reid \
          --loss triplet \
          --preprocess resize_pad \
          --epochs 120 \
          --project runs/reid_train \
          --name lmbn_joint
        ```

        Joint training from ReID data YAMLs:

        ```bash
        boxmot train-reid \
          --model csl_tinyvit_23m \
          --data market1501.yaml \
          --data duke.yaml \
          --epochs 120 \
          --device 0
        ```

        Train from a BoxMOT ReID config:

        ```bash
        boxmot train-reid --cfg custom_config.yaml
        ```

        The training-only GlobalAP, human-pretraining, and privileged-teacher
        pipeline for the 7M V20 model is documented in
        [CSL-TinyViT-7M HP-GRD](../guides/csl-tinyvit-7m-hpgrd.md).

        Train the promoted CSL-TinyViT-11M V20 preset on Market1501:

        ```bash
        MARKET1501_DIR=/data/Market-1501-v15.09.15 \
          ./train_csl_tinyvit_11m_v20.sh
        ```

        The equivalent direct command is:

        ```bash
        uv run python -m boxmot.engine.cli train-reid \
          --recipe csl_tinyvit_11m \
          --model csl_tinyvit_11m_v20 \
          --data-dir /data/Market-1501-v15.09.15 \
          --device mps \
          --num-workers 4 \
          --project runs/csl_tinyvit_11m_v20 \
          --name market1501_seed0
        ```

        This is the validated RGB-only semantic-fine recipe, without Hi-AFA or
        multilevel-suppression experiments. It starts a fresh run from the
        checksum-verified official TinyViT-11M weights; use `--resume` only for
        an existing compatible checkpoint. The canonical recipe uses four data
        workers on MPS. The launcher validates all three dataset splits and lets
        you override that default with `CSL_TINYVIT_11M_NUM_WORKERS`.

        The best recorded 11M run is not that RGB-only training policy. It is
        the A11v8 multiscale EMA pose-teacher treatment (91.02% mAP and 95.90%
        rank-1 at epoch 190). Train its current-code, checkpoint-safe equivalent
        with:

        ```bash
        MARKET1501_DIR=/data/Market-1501-v15.09.15 \
        PAV_METADATA_DIR=/data/Market-1501-pav-metadata-clean \
          ./train_csl_tinyvit_11m_v20_pose_teacher.sh
        ```

        The equivalent direct command is:

        ```bash
        uv run python -m boxmot.engine.cli train-reid \
          --recipe csl_tinyvit_11m_v20_pose_teacher \
          --model csl_tinyvit_11m_v20 \
          --data-dir /data/Market-1501-v15.09.15 \
          --anatomical-metadata-dir /data/Market-1501-pav-metadata-clean \
          --device mps \
          --num-workers 4 \
          --project runs/csl_tinyvit_11m_v20_pose_teacher \
          --name market1501_seed0
        ```

        Pose metadata is privileged training supervision only. Deployment
        prunes the teacher and retains the same 1536-D RGB descriptor. Start a
        fresh run from the checksum-verified official weights; the historical
        A11v8 `best.pt` is an inference checkpoint and cannot be resumed.

        Train the stabilized Hi-AFA reproduction profile on Market1501:

        ```bash
        MARKET1501_DIR=/data/Market-1501-v15.09.15 \
          ./train_hi_afa_market1501.sh
        ```

        This uses the registered `hi_afa` backbone at 384x128, PK sampling
        with 8 identities x 8 instances, and the paper's summed 17-head CE and
        5-stream multi-similarity objectives. The paper's 22-stream center term
        does not specify shared versus branch-specific centers; the stabilized
        profile disables BoxMOT's ambiguous shared-center interpretation.

        LDAM's trainable spatial and channel residual gates are initialized at
        zero, keeping the pretrained OSNet path identity-safe at startup.
        Evaluation excludes the DropBlock copy of `g4` and emits an 8192-D
        descriptor from 16 unique 512-D raw pooled streams. Each stream is L2
        normalized before concatenation and the concatenated descriptor is
        normalized once more. The config documents the remaining explicit
        defaults for details that the paper does not report. Its default run
        name is `stable_seed0`, separate from literal-paper experiments.

        Transfer the promoted 7M V20 hierarchy and training-only pose teacher
        to the MobileNetV4 Medium backbones:

        ```bash
        boxmot train-reid \
          --recipe mobilenetv4_conv_medium_v20 \
          --data-dir /data/Market-1501-v15.09.15 \
          --anatomical-metadata-dir /data/Market-1501-pav-metadata-clean

        boxmot train-reid \
          --recipe mobilenetv4_hybrid_medium_v20 \
          --data-dir /data/Market-1501-v15.09.15 \
          --anatomical-metadata-dir /data/Market-1501-pav-metadata-clean
        ```

        Both recipes use the standard 1/2/4 scale-balanced head, shared
        multiscale MCPT, and a training-only EMA anatomical teacher, while
        deploying a 1,152-D RGB-only descriptor. Their 100-epoch horizon adopts
        the matched Conv-M evidence that an 80-epoch cosine LR collapses before
        the observed epoch-70 optimum; Hybrid-M still requires its own sweep.
        The 200-epoch TinyViT phases are scaled to MCPT 10→25 with its identity
        prior removed at 35, and anatomy ramp 0→25, hold to 60, decay to 85,
        followed by 15 RGB-only consolidation epochs. Conv-M promotes the
        resolution-matched
        `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k` checkpoint,
        `spatial_linear` C5 projection, stride-16 final map, normal MobileNet
        augmentation, backbone LR multiplier 1.0, weight decay `1e-4`, and the
        CNN ReID neck. Hybrid-M retains the stable ix/e550/r256 MQA pretrained
        family and its lower fine-tuning LR.

        The generic MobileNetV4 default is `--timm-head-mode pooled`, matching
        timm's classification path by globally pooling C5 before its pretrained
        head. The promoted Conv-M V20 recipe overrides it with
        `spatial_linear`. The available modes are:

        - `spatial` retains the C5 map through the complete pretrained head.
        - `spatial_adapt_norm` also updates the head normalization during the
          frozen-backbone warm-start.
        - `spatial_linear` retains the pretrained 1x1 projection but bypasses
          its pooled-domain normalization and activation.
        - `off` bypasses the classification projection and uses raw C5.

        The completed `ablation_mobilenetv4_medium_v20_timm_head.sh` and
        `ablation_mobilenetv4_medium_v20_next.sh` scripts preserve the original
        pooled/spatial and optimization studies. Their results rejected strong
        augmentation and a 0.25 backbone LR multiplier, found `5e-4` weight
        decay neutral, and promoted the r384 checkpoint. The matched 100-epoch
        stride-16-map run reached 86.71% mAP versus 86.18% for its 80-epoch
        counterpart; the shorter cosine horizon was already over-decayed at
        epoch 70. The spatial LayerNorm neck remains an opt-in experiment.
        `--mobilenetv4-last-stride 1` changes the final map from 12x4 to 24x8
        at 384x128.

        A saved run can be reproduced with the same path:

        ```bash
        boxmot train-reid --cfg runs/my_experiment/hparams.json --name reproduced
        ```

        Saved `hparams.json` files use resume-compatible legacy normalization,
        so fields introduced after an older run retain their historical
        disabled behavior.

        Explicit CLI flags override the config:

        ```bash
        boxmot train-reid --cfg custom_config.yaml --epochs 3
        ```

        Example `market1501.yaml`:

        ```yaml
        dataset: market1501
        path: ../datasets/Market-1501-v15.09.15
        train: bounding_box_train
        query: query
        gallery: bounding_box_test
        download: |
          from pathlib import Path
          Path(yaml["path"]).mkdir(parents=True, exist_ok=True)
        ```

## Core idea

`train-reid` builds a ReID backbone, loads one or more registered ReID datasets, and optimizes the model with either softmax or triplet-style training.

The crop preprocessing you choose here should match the preprocessing used later at inference time.

## Identity-preserving background mosaic

Background mosaic keeps the complete anchor person and their detected backpack,
handbag, or suitcase while replacing only the surrounding background with four
donor-background tiles. The anchor PID remains the sole training label, and the
augmentation is never applied to query or gallery evaluation images.

Generate dedicated high-confidence masks for the training split first:

```bash
uv run python -m tools.create_market1501_person_masks \
  --source Market-1501-v15.09.15 \
  --output Market-1501-mosaic-highconf \
  --model weights/yolo26x-seg.pt \
  --device mps \
  --batch-size 16 \
  --conf 0.50 \
  --masks-only
```

Train on the original images while pointing the augmentation at the generated
masks:

```bash
boxmot train-reid \
  --cfg boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml \
  --data-dir Market-1501-v15.09.15 \
  --background-mosaic \
  --background-mosaic-mask-dir Market-1501-mosaic-highconf-masks \
  --background-mosaic-probability 0.30 \
  --device mps \
  --project runs/csl_tinyvit_11m_market1501_mosaic \
  --name a11s2_background_mosaic
```

The default schedule leaves mosaic disabled through epoch 10 and linearly
ramps its probability to 0.30 at epoch 30. Masks retaining less than 20% or
more than 90% of an image, masks missing the central crop region, and missing
masks all fall back to the unmodified anchor image. The generated mask root
contains `primary/` masks for preserving only the labeled person and their
nearby bags, plus `all_people/` masks that remove every high-confidence person
and nearby bag from donor tiles.

## Cross-camera same-ID part mosaic

Use `--same-id-part-mosaic` to replace one or two body-aligned regions with
corresponding regions from independently augmented images of the same identity
in the current P×K batch. Different-camera donors are preferred when available,
the hard identity label is unchanged, and evaluation images are never modified.

```bash
boxmot train-reid \
  --cfg boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml \
  --data-dir Market-1501-v15.09.15 \
  --same-id-part-mosaic \
  --same-id-part-mosaic-probability 0.35 \
  --same-id-part-mosaic-max-regions 2 \
  --same-id-part-mosaic-min-area 0.15 \
  --same-id-part-mosaic-max-area 0.40 \
  --same-id-part-mosaic-boundary-jitter 0.05 \
  --same-id-part-mosaic-cross-camera-rate 1.0 \
  --same-id-part-mosaic-min-unaltered 0.5 \
  --device mps \
  --project runs/csl_tinyvit_11m_market1501_sameid_mosaic \
  --name a11s2_sameid_partmosaic_p035
```

The default policy replaces 15–40% of selected images, jitters anatomical
boundaries by up to 5% of image height, and leaves at least half of each batch
unaltered. It needs no segmentation masks or mixed labels. The camera-aware
sampler is complementary because it makes cross-camera same-ID donors available
more consistently. When enabled, the existing Random Erasing policy is applied
independently after the part composite.

## Pose-aligned view mosaic

PAV-Mosaic uses YOLO pose keypoints to replace semantic body parts instead of
horizontal rectangles. Head, torso, left/right arms, upper/lower legs, and
nearby bags are selected from high-confidence same-ID observations, preferably
from other cameras and poses, then warped into the anchor geometry. The anchor
background is unchanged unless context mosaic is also enabled.

Generate the training-only metadata first. YOLO26x-seg supplies the person
foreground and separate bag masks used to constrain the pose-derived regions:

```bash
uv run python -m tools.create_market1501_pav_metadata \
  --source Market-1501-v15.09.15 \
  --output Market-1501-pav-metadata \
  --pose-model https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt \
  --seg-model https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt \
  --device mps \
  --batch-size 16 \
  --pose-conf 0.25 \
  --seg-conf 0.50
```

Then train the PAV-only arm:

```bash
boxmot train-reid \
  --cfg boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml \
  --data-dir Market-1501-v15.09.15 \
  --pav-mosaic \
  --pav-metadata-dir Market-1501-pav-metadata \
  --pav-mosaic-probability 0.25 \
  --pav-mosaic-max-parts 3 \
  --pav-mosaic-max-foreground-replacement 0.45 \
  --pav-mosaic-cross-camera-rate 0.8 \
  --pav-mosaic-different-pose-rate 0.5 \
  --pav-mosaic-min-keypoint-confidence 0.5 \
  --pav-mosaic-min-unaltered 0.5 \
  --pav-mosaic-warmup-epochs 40 \
  --pav-mosaic-decay-start-epoch 170 \
  --pav-mosaic-final-probability-scale 0.5 \
  --device mps \
  --name a11u6_pavmosaic
```

The context arms additionally enable the existing foreground-preserving
background mosaic at probability 0.20. A real, high-confidence donor person can
enter from the left, right, or bottom boundary at probability 0.15 and cover
5–20% of the image without contributing its identity to the label. The final
arm sets `--pav-consistency-weight 0.2`; it applies ID loss to both the clean and
mosaic observations and cosine consistency to their retrieval descriptors.
Only successfully augmented pairs incur the extra clean forward pass.

The complete controlled comparison is in
`ablation_csl_tinyvit_11m_a11u_sampler_mosaic.sh`:

- `a11u4`: same-ID rectangular part mosaic.
- `a11u6`: PAV-Mosaic.
- `a11u7`: PAV-Mosaic plus context mosaic and realistic occluders.
- `a11u8`: PAV/context plus clean-view consistency.

## Privileged anatomical supervision

`--anatomical-auxiliary` uses the same pose/person-mask metadata to supervise
six ordered RGB tokens: head, torso, left/right arms, and left/right legs. A
deterministic pose-mask router defines the spatial target for every cell, while
learned RGB queries receive same-scale token consistency, attention KL,
visibility, dense geometry, and optional same-part cross-camera contrastive
losses. There is no learned pose encoder or second backbone forward, and the
normal global/stripe descriptor is unchanged at evaluation and export.

The default `--anatomical-target-type deterministic_scale_aware_geometry`
selects this deterministic router. To reproduce the A11v8 teacher, use
`--anatomical-target-type learned_pose_concat_ema` together with
`--anatomical-teacher-momentum 0.999`. That path restores the learned
pose-heatmap encoder, the fine-map online teacher, and its stop-gradient EMA
copy. It supervises both the local and fine anatomical students during
training, but is still omitted from evaluation and export. Legacy A11v8
hyperparameters that contain a teacher momentum but no target type are mapped
to this path automatically.

Target routing uses an actual pose-aligned `4x2` grid for every part.
The torso grid is mapped through the shoulder/hip quadrilateral, limb rows
follow the shoulder-elbow-wrist or hip-knee-ankle chain from proximal to distal,
and the head grid is oriented by bilateral eye/ear landmarks. Each valid cell
creates a normalized spatial routing distribution, clipped by its cleaned
person-part mask when one is available. Its stop-gradient, same-scale RGB
feature average is the token target. Every grid cell owns a slice of the token
channels and a separately supervised RGB-student attention map, preserving
anatomical layout instead of reducing it to an unordered image-space average.

Set `--anatomical-descriptor-distill-weight` above zero to enable the stronger
descriptor path. The six visibility-weighted local semantic tokens form a
geometry-routed descriptor that supervises the deployed global/stripe
descriptor with both cosine alignment and pairwise-similarity distillation.
The comparison projection is training-only; inference input, descriptor shape,
and compute remain unchanged.

Set `--anatomical-pose-teacher-weight` above zero for dense geometric coverage.
Transformed COCO-17 grids and cleaned person-part masks deterministically define
where each anatomical cell must attend. Local targets cover a broader fraction
of the person to match the semantic map's receptive field; fine targets are
sharper and retain limb and boundary detail. RGB is never concatenated with
pose, so the model cannot bypass the privileged signal through an RGB-only
teacher projection. Pose is not passed through the model and is never required
by evaluation or export.

Person-mask-validated grid cells take priority whenever a mask exists. Pose-only
records can remain usable at reduced reliability, though cleaned metadata can
set this reliability to zero. Geometry-routed feature averages provide
same-scale token-consistency targets, and all anatomical losses can still be
ramped and removed before the final retrieval-only epochs.
At startup, the trainer verifies that metadata records match the selected
training images and that declared person-mask files exist. Usable coverage
counts only confidence-qualified poses and, when pose-only reliability is
zero, readable nonempty masks; it must meet
`--anatomical-min-effective-coverage`. The resume
contract fingerprints the metadata manifest and referenced mask bytes, so
annotation drift cannot silently enter an exact continuation. Routing,
normalization, and anatomical KL/contrastive calculations run in FP32 even
when the RGB model uses CUDA mixed precision, preventing empty-cell underflow
from contaminating a batch.

```bash
boxmot train-reid \
  --cfg boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml \
  --data-dir Market-1501-v15.09.15 \
  --anatomical-auxiliary \
  --anatomical-target-type deterministic_scale_aware_geometry \
  --anatomical-metadata-dir Market-1501-pav-metadata \
  --anatomical-token-dim 128 \
  --anatomical-distill-weight 0.10 \
  --anatomical-attention-weight 0.10 \
  --anatomical-visibility-weight 0.05 \
  --anatomical-contrastive-weight 0.10 \
  --anatomical-descriptor-distill-weight 0 \
  --anatomical-pose-teacher-weight 0.03 \
  --anatomical-pose-only-reliability 0.35 \
  --anatomical-min-effective-coverage 0.8 \
  --anatomical-student-start-epoch 20 \
  --anatomical-student-ramp-end-epoch 50 \
  --anatomical-decay-start-epoch 120 \
  --anatomical-decay-end-epoch 170 \
  --anatomical-temperature 0.07
```

Masks, pose grids, and keypoints follow resizing, random translation, horizontal
flips, RandomPatch, and Random Erasing. Horizontal flips also exchange left/right
token labels. A sample with valid cached pose but no person mask can still train
the RGB student at the configured pose-only reliability; mask-dependent
visibility supervision is skipped. A sample without valid geometry remains
ordinary RGB-only ReID training.
When both estimates exist, mask-validated cells define the target and low
pose/person-mask agreement reduces its anatomical weight. Sparse targets are
compressed in memory after their first construction, and metrics report each anatomical loss component,
usable part coverage, and cross-camera-positive coverage separately.

Enable `--anatomical-multiscale` to mirror the hierarchical stripe routing.
The existing Stage-2/local anatomical student and a new Stage-0/fine student
share a role basis, while explicit cell embeddings, scale-specific query
offsets, projections, and normalization let each resolution specialize. Each
student learns from geometry and RGB targets constructed at its own resolution.
Their complete token-consistency, attention, visibility, geometry, and
cross-camera contrastive losses are balanced with
`--anatomical-local-scale-weight` and
`--anatomical-fine-scale-weight`, which must sum to one. Corresponding
fine/local tokens align only their within-image role-similarity structure using
`--anatomical-cross-scale-weight`; raw tokens remain free to encode
scale-specific information. These branches remain training-only.
By default the fine-map and cross-scale terms follow the shared anatomical
student schedule. Set `--anatomical-fine-start-epoch` and
`--anatomical-fine-ramp-end-epoch` to introduce those terms later while the
Stage-2/local student retains the shared schedule. The cross-scale term follows
the fine-map ramp.

Select `--anatomical-target-type privileged_mask_pose_attention` to use pose
and masks strictly as targets for an RGB attention adapter. The adapter predicts
person foreground and six soft anatomical maps at the Stage-2 and Stage-0
resolutions. A bounded residual gate modifies the local and fine RGB maps before
the existing fixed-stripe pooling. Mask foreground and pose-part evidence have
independent learned strengths, so retrieval can retain either cue if the other
is noisy. Both strengths are initialized to zero, share a bounded residual
budget, and remain disabled during the configured backbone-freeze epochs. The
global branch and the 1536-D retrieval contract are unchanged.

Use `--anatomical-person-mask-dir` to supply the external high-confidence masks.
They take priority over masks referenced by the pose metadata. Pose-only images
train the six part maps at reduced reliability, mask-only images train the
foreground map, and images with neither annotation retain the ordinary ReID
loss. `--anatomical-foreground-weight` controls the BCE/Dice foreground
objective. A11v8's same-scale token consistency remains enabled, now using
stop-gradient RGB averages routed by the pose/mask regions. Descriptor and
stripe-branch distillation should remain disabled for this target type.

The supplied training script inherits the complete A11v8 control configuration,
uses cleaned pose metadata with pose-only reliability set to zero, and preserves
the successful multi-scale supervision schedule: ramp to full strength by
epoch 50, decay from epochs 120 through 170, then finish with 30 retrieval-only
epochs. Its startup preflight rejects a control run whose architecture,
sampler, optimizer, schedule, or recorded best result no longer matches A11v8.

At evaluation and export, the adapter predicts its gate from RGB features.
No metadata, mask, pose tensor, pose estimator, or segmentation estimator is
loaded. The complete Market-1501 command is provided by
`train_csl_tinyvit_11m_privileged_mask_pose_attention.sh`.

Use `learned_pose_semantic_ema` to keep the complete A11v8 retrieval path and
add local/fine foreground and six-part prediction heads strictly during
training. `learned_pose_semantic_fused_ema` additionally blends each pose-cell
attention target with its person-mask-clipped part mask. The blend is computed
per image and part: weak pose confidence shifts weight toward the semantic
mask, while low pose-mask agreement shifts it back toward pose geometry.
Neither mode instantiates the privileged residual gate or changes the 1536-D
inference descriptor.

`--anatomical-foreground-weight` and
`--anatomical-semantic-part-weight` control the training-only foreground and
six-part BCE/Dice objectives. Current Market-1501 PAV metadata provides
pose-routed part masks clipped by a high-confidence person silhouette; it is
not independent human parsing for garment or shoe classes. The prediction-head
interface can consume richer six-role parsing masks when such annotations are
available.

The controlled A11v18 sequence is in
`ablation_csl_tinyvit_11m_pose_semantic_teacher.sh`: a current-code A11v8
replication, two channel-representation controls, semantic losses without
target fusion, confidence-fused targets, then coarse/fine-only branch
distillation at weight 0.025. All arms preserve A11v8's multi-scale weights
and supervision schedule. `stage2_channel2` appends two shared 128-D Stage-2
channel specialists and evaluates a 1792-D descriptor.
`multiscale_channel2` instead adds two 128-D channel summaries to each of the
global, coarse, and fine source maps. With
`--multiscale-channel-alpha 0.5`, every scale assigns 75% of its descriptor
power to spatial branches and 25% to its channel pair; final normalization
still assigns one-third total power to each scale. Its raw metric and deployed
BN descriptors both include all six summaries and have 2304 dimensions. The
three scale-specific projections and shared-within-scale BNNecks add 388,224
parameters for Market-1501. The pose/semantic-only arms retain the 1536-D
descriptor.

For a fully decoupled parsing treatment, select
`decoupled_pose_parsing_teacher`. Private local/fine parsing adapters receive
foreground and dense part supervision, and parsing masks restrict a shared
set of teacher queries. Corresponding RGB queries attend without masks and
learn from stop-gradient teacher tokens. Query diversity and visible-part hard
triplet losses are controlled by `--anatomical-query-diversity-weight` and
`--anatomical-part-triplet-weight`; their ramp has separate query start/end
options. `--anatomical-accessory-query` adds an optional seventh bag query
whose reliability is zero when no bag mask exists.

`--anatomical-query-relational-distill-weight` additionally matches the
student and masked teacher cosine-similarity matrices separately for each
semantic query. Only reliable cross-camera pairs contribute, and same-ID and
different-ID pair groups are balanced before averaging. This transfers
identity geometry without adding a deployed branch.

`--clean-student-consistency-weight` creates a deterministic resized clean
teacher view alongside the normal augmented RGB student. Clean masked teacher
queries supervise the augmented unrestricted queries, while the detached clean
retrieval descriptor provides view consistency. This treatment is mutually
exclusive with PAV mosaic consistency and does not add clean-view ID loss.

Both query paths are training-only. The local/fine retrieval maps are never
gated or replaced, and evaluation keeps A11v8's 1536-D RGB descriptor. Run the
four controlled arms with
`ablation_csl_tinyvit_11m_decoupled_pose_parsing_teacher.sh`.
The gated V20 control, relational-query, clean-student, and branch-only suite
is in `ablation_csl_tinyvit_11m_pose_distillation_v20.sh`.

The promoted V8 pose-teacher policy has both an explicit RGB architecture preset
and a width-adapted training recipe named `csl_tinyvit_7m_v20`. Direct preset
construction does not require pose or mask metadata and has 6,937,893 parameters
for 751 training identities. The recipe opts into a 227,118-parameter privileged
anatomy teacher, bringing its training model to 7,165,011 parameters; that branch
is absent from the deployed RGB model. Both retain the 384×128 global/two-stripe/four-stripe
hierarchy, rectangular attention, scale-balanced descriptor, PK sampling, and
pose-loss schedule. The fusion/retrieval width is reduced from 512 to 384 and
the anatomical bottleneck from 128 to 96 to match the 7M backbone's
64/128/160/320 stage widths. Consequently, global, coarse, and fine scales
contribute 384 dimensions each to a 1152-D deployed RGB descriptor. Run the
recommended model with `ablation_csl_tinyvit_7m_v20_transfer.sh`; optional
`rgb` and `unscaled` arms provide matched controls.

The controlled Hi-AFA-lite treatment is the
`csl_tinyvit_7m_hi_afa_lite` recipe. It is identical to
`csl_tinyvit_7m_v20` except that a reduction-4 ReID adapter is enabled at
Stage 3 and its lateral input uses feature-selective suppression with
`tau=0.7`. The main TinyViT stream remains dense, the adapter gate remains
zero-initialized, and evaluation still emits the standard 1152-D RGB
descriptor. Because the V20 pose teacher is preserved, training also requires
the generated PAV metadata:

```bash
MARKET1501_DIR=/data/Market-1501-v15.09.15 \
PAV_METADATA_DIR=/data/Market-1501-pav-metadata-clean \
  ./train_csl_tinyvit_7m_hi_afa_lite.sh
```

The equivalent direct command is:

```bash
uv run python -m boxmot.engine.cli train-reid \
  --recipe csl_tinyvit_7m_hi_afa_lite \
  --data-dir /data/Market-1501-v15.09.15 \
  --anatomical-metadata-dir /data/Market-1501-pav-metadata-clean
```

The training-only multilevel classifier-guided suppression treatment is
available as `csl_tinyvit_7m_multilevel_suppression`. It is an exact V20
feature-evidence ablation inspired by Hi-AFA, not a paper-faithful Hi-AFA
implementation. Stage-3 ReID adapters remain disabled and the deployed model
uses a 7,165,011-parameter training graph and the same pruned 1152-D RGB
descriptor at deployment. During
training, a detached target-class Grad-CAM from a frozen copy of the global
classifier masks the strongest locations independently in each stripe of a
private coarse-map copy. The frozen scorer uses accumulated BN running
statistics, so each image's CAM is independent of the other identities in its
batch. The two coarse classifiers then produce separate CAMs for their own
halves; those CAMs are stitched and resized to guide a private four-stripe
fine-map copy. Stripes with a constant or invalid CAM are left intact and are
excluded from the auxiliary CE loss. The clean global, coarse, and fine maps,
logits, descriptor, and BN statistics are never modified.

This corrected activity-masked objective is implementation version 2. Its
version is stored in hparams and resumable checkpoints, so an older
multilevel-suppression run cannot be resumed under different loss semantics.
The launcher therefore defaults to the distinct run name
`class_cam_q15_v2_seed0`.

Suppression starts at epoch 20, ramps to a loss weight of 0.2 and a ratio of
0.15 by epoch 50, begins decaying at epoch 140, and is disabled after epoch
170. The ratio is rounded up to a whole spatial location within each stripe,
so the recorded erase fraction can be slightly above the requested value.
Because the underlying V20 pose-teacher recipe is unchanged, the launcher
requires the same PAV metadata:

```bash
MARKET1501_DIR=/data/Market-1501-v15.09.15 \
PAV_METADATA_DIR=/data/Market-1501-pav-metadata-clean \
  ./train_csl_tinyvit_7m_multilevel_suppression.sh
```

To customize the output without editing the recipe, set
`MULTILEVEL_SUPPRESSION_PROJECT`, `MULTILEVEL_SUPPRESSION_NAME`,
`MULTILEVEL_SUPPRESSION_DEVICE`, or `MULTILEVEL_SUPPRESSION_NUM_WORKERS`.
Set `VALIDATE_ONLY=1` to check the two input roots and print the resolved
command without starting training. Epoch metrics record the scheduled ratio,
actual coarse/fine erase fractions, and both CAM-active fractions so degenerate
or ineffective masks are visible during the run.

The focused `ablation_csl_tinyvit_7m_mcpt_pose.sh` study combines this
training-only V8 teacher with shared-multiscale MCPT in a same-source 2×2
factorial: RGB, MCPT, pose, and MCPT plus pose. The combination is intentionally
limited to the 7M standard scale-balanced stripe head and the multiscale
`learned_pose_concat_ema` teacher. Evaluation still requires RGB only and emits
the same 1152-D descriptor; MCPT remains active while the pose branch is
discarded. Optional `foreground` and `foreground_combo` arms compare the
foreground-aware MCPT alternative.

The corresponding larger-backbone recipe is `csl_tinyvit_23m_v8`. It keeps
the same V8 policy while scaling the retrieval width to 640 and the anatomical
bottleneck to 160 for the 23M backbone's 96/192/384/576 stage widths. Global,
two-stripe, and four-stripe scales therefore contribute 640 dimensions each to
the 1920-D deployed RGB descriptor. The 23M backbone retains its native 0.20
DropPath rate. Run the matched RGB and pose-teacher treatments with
`ablation_csl_tinyvit_23m_v8_transfer.sh`.

For a deployed slot representation, select
`--head-type body_slot --anatomical-target-type body_slot_privileged_ema`.
This replaces the fixed global/two-stripe/four-stripe head with a 512-D global
descriptor and eight persistent 128-D RGB slots. The same slots read Stage 0,
Stage 2, and Stage 3, so the final normalized descriptor remains 1536-D.
Visibility controls descriptor power within the slot stream, while
`--body-slot-alpha` controls the global-versus-slot split.

The recommended `--body-slot-mode recurrent_read` never changes backbone
features. `recurrent_read_write` adds slot-to-spatial attention at all three
stages, with every residual gate initialized to exactly zero. With 751
Market-1501 classes, the read-only replacement has 13,522,022 parameters
versus 13,514,597 for the stripe model; read/write has 13,853,289.

Pose-derived parts, the person mask, and the accessory mask form weak teacher
roles only during training. Masked EMA projections supervise slot embeddings,
attention, visibility, diversity, foreground coverage, and visible
cross-camera part triplets. Evaluation, export, and tracking receive RGB
images only. The controlled Tier B command and opt-in Tier C arm are in
`ablation_csl_tinyvit_11m_body_slots.sh`.

Enable `--anatomical-deployment` to make the six pose-supervised RGB students
part of the retrieval descriptor. Local and fine tokens for head, torso,
left/right arms, and left/right legs are fused and reduced to
`--anatomical-deployment-dim` channels each. Their RGB visibility predictions
weight the normalized part descriptor, which is appended to the unchanged base
descriptor with relative energy `--anatomical-deployment-alpha`.

Only the training teacher consumes pose. Evaluation, export, and tracking run
the RGB student path without keypoints. With the 1536-D hierarchical base,
six 64-D parts produce a 1920-D descriptor. Visibility-weighted part-ID and
cross-camera metric losses remain active after scheduled teacher losses decay;
their weights are controlled by `--anatomical-deployment-id-weight` and
`--anatomical-deployment-metric-weight`.

This treatment requires the learned EMA pose teacher, multi-scale anatomy, and
`norm_concat_bn`. It is intentionally incompatible with descriptor
distillation, stripe branch distillation, and the compact deployment head so
each experiment has one deployed representation treatment.

```bash
boxmot train-reid \
  --cfg boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml \
  --data-dir Market-1501-v15.09.15 \
  --anatomical-auxiliary \
  --anatomical-target-type learned_pose_concat_ema \
  --anatomical-metadata-dir Market-1501-pav-metadata-clean \
  --anatomical-multiscale \
  --anatomical-deployment \
  --anatomical-deployment-dim 64 \
  --anatomical-deployment-alpha 0.25 \
  --anatomical-deployment-id-weight 0.25 \
  --anatomical-deployment-metric-weight 0.10 \
  --anatomical-descriptor-distill-weight 0 \
  --anatomical-branch-distill-weight 0
```

For A11v13-style branch-aligned EMA distillation, set
`--anatomical-branch-distill-weight` above zero with
`--anatomical-target-type learned_pose_concat_ema`. The stop-gradient EMA
teacher softly assigns reliable canonical cells to the deployed global,
two-stripe, and four-stripe descriptors, then matches same-ID and different-ID
cross-camera relations at each level. The global, coarse, and fine
coefficients must sum to one. This path requires the standard scale-balanced
hierarchical stripe head and `norm_concat_bn`, and adds no inference output,
parameters, or latency.

```bash
boxmot train-reid \
  --cfg boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml \
  --data-dir Market-1501-v15.09.15 \
  --anatomical-auxiliary \
  --anatomical-target-type learned_pose_concat_ema \
  --anatomical-metadata-dir Market-1501-pav-metadata-clean \
  --anatomical-pose-teacher-weight 0.03 \
  --anatomical-multiscale \
  --anatomical-local-scale-weight 0.60 \
  --anatomical-fine-scale-weight 0.40 \
  --anatomical-cross-scale-weight 0.05 \
  --anatomical-branch-distill-weight 0.05 \
  --anatomical-branch-global-coefficient 0.20 \
  --anatomical-branch-coarse-coefficient 0.30 \
  --anatomical-branch-fine-coefficient 0.50 \
  --anatomical-fine-start-epoch 40 \
  --anatomical-fine-ramp-end-epoch 80
```

## Width-first hierarchy and identity registers

`--width-first-hierarchy` changes CSL-TinyViT's early spatial allocation from
the usual joint height/width reduction to `48x16 -> 48x8 -> 24x8`. Stage 1
therefore models the full pedestrian height with alternating `12x4` and
`16x4` windows before a height-only merge. The intended compute-balanced
setting moves one block from Stage 2 to Stage 3 with
`--stage2-depth 5 --stage3-depth 3`.

`--identity-registers` adds four recurrent global tokens after Stage 2 and
Stage 3. Each register reads window summaries, and the resulting context is
broadcast back to the spatial map through a scalar zero-initialized gate.
Window summaries are projected from 448 dimensions into the
`--identity-register-dim 128` communication bottleneck, then projected back
to the 448-D spatial map. The two communication modules therefore add about
0.63M parameters rather than the roughly 5.24M required by full-width
registers.
`--identity-register-dropout` drops whole registers during training, while
`--identity-register-diversity-weight 0.01` weakly discourages duplicated
registers. The registers require the unchanged standard global/2-stripe/
4-stripe head. They are treated as ReID adaptation parameters, so their
randomly initialized communication modules train at head LR during backbone
warm-start; the learned register seed is not weight-decayed. They are internal
backbone state: evaluation still emits the same RGB-only 1536-D
`norm_concat_bn` descriptor.

The controlled A11v8-based register treatment is:

```bash
./ablation_csl_tinyvit_11m_reid_x.sh
```

The default `a11x2r_v8_four_identity_registers_d128` arm changes only the
register path and retains A11v8's Stage-2/3 depths 6/2 and normal hierarchy.
The earlier compound x2 arm used the unvalidated width-first 5/3 backbone and
collapsed to 15.88% mAP while its register gates remained near zero; it was
not a valid register ablation. Use `INCLUDE_WIDTH_FIRST=1` only to retain that
known-negative x1 diagnostic. Use `INCLUDE_CONTROL=1` to add a current-source
A11v8 replication or `DRY_RUN=1` to print and preflight commands without
training.

## Fixed camera-aware PK sampling

Use `--camera-aware-sampler` to draw one image from each available camera
before adding same-camera instances for an identity. `--pk-steps-per-epoch`
fixes the number of training batches so changing K does not silently shorten
an epoch.

The batch-96 P16K6 procedure matching the promoted P12K8 training budget is:

```bash
boxmot train-reid \
  --cfg boxmot/reid/training/configs/recipes/csl_tinyvit_11m.yaml \
  --data-dir Market-1501-v15.09.15 \
  --p-ids 16 \
  --k-instances 6 \
  --pk-steps-per-epoch 62 \
  --camera-aware-sampler \
  --project runs/csl_tinyvit_11m_market1501_sampler \
  --name a11s2_p16k6_camera_aware_s62
```

This produces exactly 62 batches and 5,952 image draws per epoch, matching the
current P12K8 procedure while increasing the identity count per batch.

## Modular CSL-TinyViT ablations

CSL-TinyViT training resolves the flat CLI/config values into a canonical
ablation plan before model construction. The plan separates five independent
axes:

- architecture
- retrieval head
- augmentation
- privileged supervision
- auxiliary objective

The selected head and every enabled treatment are written to
`model.ablation` in `hparams.json` and checkpoint metadata. This makes reports
show the actual treatment set instead of requiring experiment names to encode
it. Existing commands remain unchanged; for example,
`--head-type multiscale_channel2 --pav-mosaic --csmm-loss-weight 0.2` resolves
to one head, one augmentation, and one objective.

Implementation registries are intentionally separated by responsibility:

- `boxmot.reid.backbones.head_registry` defines supported heads and their
  capabilities once for the CLI, trainer, and models.
- `boxmot.reid.backbones.option_registry` defines categorical mode choices
  such as feature fusion, pooling, and descriptor selection.
- `boxmot.reid.training.ablation` defines named optional treatments,
  dependencies, and mutually exclusive groups.
- `boxmot.reid.training.model_options` groups trainer-to-model arguments by
  component.
- `boxmot.reid.training.augmentations` validates and assembles the image,
  sample-level, clean-view, and privileged-target pipeline.

When adding an ablation, add one named treatment to the registry, put its model
arguments in the corresponding option group, and add a focused component test.
Avoid creating a second boolean for a mode that is already represented by
`head_type`, `feature_fusion`, or another exclusive selector.

## Supported datasets

The built-in dataset registry currently includes common ReID benchmarks such as:

- `market1501`
- `duke` / `dukemtmcreid`
- `cuhk03`
- `msmt17`
- `msmt17_merged`

You pass the dataset root through `--data-dir`, and BoxMOT resolves the expected subdirectory layout for the selected dataset.

Alternatively, pass one or more `--data` YAML configs. YAML `path` values are resolved relative to the YAML file, and `download` is a local Python block executed only when that root is missing or empty. Built-in ReID datasets still use their registered parsers; `train`, `query`, and `gallery` are saved in hparams as dataset metadata.

## Main outputs

Training writes an experiment directory under `--project/--name`, typically containing:

- best and last checkpoints
- training logs and metrics
- periodic validation results

When training finishes, BoxMOT reports the best checkpoint path along with the best validation `mAP` and `rank-1` score.

## Resuming and evaluation during training

- Use `--resume` with a checkpoint directory or `last.pt` file to continue an interrupted run.
- Use `--eval-interval` to control how often validation runs during training.
- Use `--eval-datasets` for extra cross-domain checks during training.

## Scope

The CLI command is `train-reid`; the same workflow is available through the
high-level `BoxMOT.train(...)` Python facade.

```python
from boxmot import BoxMOT

model = BoxMOT("mobilenetv4")
model.train(cfg="mobilenetv4_custom.yaml")
```

When the first positional argument matches a registered ReID training recipe or backbone, it is used as the training profile; detector names still configure tracking detectors. A ReID weight filename can also seed the training profile while binding the object to that weight for later export or embedding:

```python
reid = BoxMOT(reid="mobilenetv4.pt")
reid.train(cfg="custom_config.yaml")
```

## Related pages

- [Evaluate ReID](eval-reid.md)
- [Export](export.md)
- [ReID Profiles](../config/reid.md)

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: train_reid
    :style: table
    :prog_name: boxmot train-reid
