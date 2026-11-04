# Train ReID

Use `train-reid` to fit an appearance encoder on identity-labeled person or
vehicle crops. Training uses the ReID dataset readers, not tracking experiment
YAMLs or materialized perception builds.

## Examples

Choose the CLI or Python tab for the same training workflow. Python options use
underscores in place of CLI hyphens. Explicit Python options take precedence
over recipes and config files. Keep the `__main__` guard when running a script
with data-loader workers.

Train OSNet on Market1501, naming the run so its output can be reused:

=== "CLI"

    ```bash
    boxmot train-reid \
      --model osnet_x0_25 --dataset market1501 \
      --imgsz 256,128 --device cuda:0 \
      --project runs/reid_train --name osnet_market
    ```

=== "Python"

    ```python
    from boxmot.engine.commands.reid.train import main as train_reid

    if __name__ == "__main__":
        result = train_reid(
            model="osnet_x0_25",
            dataset="market1501",
            imgsz=(256, 128),
            device="cuda:0",
            project="runs/reid_train",
            name="osnet_market",
        )
        print(result.weights_path)
    ```

With no `--data-dir`, the Market1501 resolver downloads and extracts the dataset
under `./datasets/reid`. Use `--data-dir /data/reid` for an existing dataset.
Pretrained model initialization can also download weights on the first run.

ReID training accepts PyTorch device names: `cpu`, `mps`, or `cuda:0`.
Use the full CUDA name; a bare `--device 0` does not work here. Recipes may set
MPS explicitly, so pass `--device` when running them on another platform.

Train the bundled RGB-only CSL-TinyViT-11M V20 recipe:

=== "CLI"

    ```bash
    boxmot train-reid \
      --recipe csl_tinyvit_11m \
      --data-dir /data/Market-1501-v15.09.15 \
      --device cuda:0 \
      --project runs/reid_train --name tinyvit_rgb
    ```

=== "Python"

    ```python
    from boxmot.engine.commands.reid.train import main as train_reid

    if __name__ == "__main__":
        result = train_reid(
            recipe="csl_tinyvit_11m",
            data_dir="/data/Market-1501-v15.09.15",
            device="cuda:0",
            project="runs/reid_train",
            name="tinyvit_rgb",
        )
        print(result.weights_path)
    ```

Joint training remaps identities across datasets so equal numeric IDs from two
sources do not refer to the same person:

=== "CLI"

    ```bash
    boxmot train-reid \
      --model lmbn_n --dataset market1501,duke,cuhk03 \
      --data-dir /data/reid --loss triplet --preprocess resize_pad \
      --epochs 120 --device cuda:0 \
      --project runs/reid_train --name lmbn_joint
    ```

=== "Python"

    ```python
    from boxmot.engine.commands.reid.train import main as train_reid

    if __name__ == "__main__":
        result = train_reid(
            model="lmbn_n",
            dataset="market1501,duke,cuhk03",
            data_dir="/data/reid",
            loss="triplet",
            preprocess="resize_pad",
            epochs=120,
            device="cuda:0",
            project="runs/reid_train",
            name="lmbn_joint",
        )
        print(result.weights_path)
    ```

Keep separate dataset roots with repeated `--data` YAMLs:

=== "CLI"

    ```bash
    boxmot train-reid \
      --model lmbn_n --data market1501.yaml --data duke.yaml \
      --epochs 120 --device cuda:0
    ```

=== "Python"

    ```python
    from boxmot.engine.commands.reid.train import main as train_reid

    if __name__ == "__main__":
        result = train_reid(
            model="lmbn_n",
            data=("market1501.yaml", "duke.yaml"),
            epochs=120,
            device="cuda:0",
        )
        print(result.weights_path)
    ```

Create those files first. For example, `market1501.yaml` can contain:

```yaml
dataset: market1501
path: /data/Market-1501-v15.09.15
```

The `duke.yaml` file uses `dataset: duke` and its own dataset path. Relative
`path` values resolve beside the data YAML. `train`, `val`, `query`, and
`gallery` fields are retained as metadata; they do **not** override the
registered reader's directories. An optional `download` field is executable
Python, run from the YAML directory only when its root is missing or empty.
A data YAML must therefore come from a source you trust and its download block
must actually populate the reader's expected layout.

## Supported datasets

Supply the complete dataset layout, including train, query, and gallery.
Most readers accept either the dataset folder or a shared parent containing a
recognized folder name.

| Name | Expected layout beneath the dataset folder |
| --- | --- |
| `market1501` | `bounding_box_train/`, `query/`, `bounding_box_test/`; `.jpg` names such as `0002_c1s1_000451_03.jpg` |
| `mot171501` | The same crop layout; shared roots recognize folders such as `MOT17-1501-fixed/` or `MOT17-1501/` |
| `duke` | `DukeMTMC-reID/{bounding_box_train,query,bounding_box_test}/`; `.jpg` names such as `0001_c2_f0046182.jpg` |
| `cuhk03` | `cuhk03-np/detected/{bounding_box_train,query,bounding_box_test}/`; new-protocol `.jpg` or `.png` crops |
| `msmt17` | `list_train.txt`, `list_query.txt`, `list_gallery.txt`, with images beneath `train/` and `test/` or the root |
| `msmt17_merged` | The MSMT17 layout; training combines train, query, and gallery lists |
| `veri` | `image_train/`, `image_query/`, `image_test/`; vehicle `.jpg` names such as `0001_c001_00016450_0.jpg` |

Aliases include `dukemtmcreid`, `mot17_1501`, `cuhk03np`, and `veri776`.
CUHK03 defaults to detected crops; its Python reader also accepts
`variant="labeled"`. The CLI does not expose that variant selector. Duke bare
roots must retain a recognized dataset folder name; pass the shared parent for
CUHK03's `detected/` layout.

MSMT17 list rows are `<relative_image_path> <person_id>`; camera IDs come from
the third underscore-separated filename field. `list_val.txt` is not consumed
by this reader. `msmt17_merged` includes evaluation images in training, so its
own query/gallery scores are not held-out results. Use a separate target
benchmark for cross-domain evaluation.

Only Market1501 has built-in automatic acquisition, and only when training
without `--data-dir` or `--data`. Other datasets need local payloads or a data
YAML download block. See the [dataset catalog](../datasets/index.md) for the
separate tracking dataset profiles.

## Configuration and batching

Settings resolve in this order: training defaults, selected recipe, `--cfg`
YAML/JSON, then explicit CLI flags. Some models select a default recipe
implicitly: the 11M V20 models use `csl_tinyvit_11m`, and the 7M V20 and
MobileNetV4 V20 models use their matching recipes. Inspect the saved
`hparams.json` for the resolved configuration.

```bash
boxmot train-reid --cfg custom_config.yaml --epochs 120 --device cuda:0
boxmot train-reid --cfg runs/reid_train/osnet_market/hparams.json --name reproduced
```

The second command starts a new run with saved settings. Use `--resume` for
an exact continuation. ReID runtime profiles under `boxmot/configs/reid`
select inference weights and are not training configs.

Training batch size is `--p-ids × --k-instances`; `--batch-size` controls
query/gallery evaluation batches. For example, `--p-ids 16 --k-instances 4`
uses 64 training images. `--source-balance 'market1501:8,4;mot17_1501:8,4'`
instead sets a separate identity quota per source. All selected sources must
exist under their configured roots.

`--camera-aware-sampler` prefers different cameras within each identity.
`--pk-steps-per-epoch` fixes the number of batches; zero uses a shuffled
identity pass. Fix this value when comparing different P/K settings so the
number of optimizer steps remains comparable.

The supported metric losses are `triplet`, `wrt`, `circle`, and `ms`; `softmax`
uses classification only. Classification supports CE, ArcFace, and CosFace.
Keep `--imgsz` and `--preprocess` consistent with later evaluation and tracking.

## Outputs and resume

The default run root is `runs/reid_train/exp`; `--project` and `--name` select
another location. Preserve these files together:

| File | Purpose |
| --- | --- |
| `best.pt` | Compact inference checkpoint selected by the primary query/gallery mAP; use for evaluation, tracking, and export |
| `last.pt` | Resumable training state, including optimizer and scheduler state |
| `hparams.json` | Resolved configuration, model settings, and resume provenance |
| `metrics.json` | Training and validation history |

```bash
boxmot train-reid --resume runs/reid_train/osnet_market
```

A directory resolves to its `last.pt`. `best.pt` is inference-only and cannot
be resumed. Exact resume validates the saved training contract and history;
retain the sidecars and the same dataset/privileged annotations. Changing the
architecture, sampling, or other contract fields is a new experiment.

Validation runs every `--eval-interval` epochs and at the final epoch. For
joint training, the first dataset supplies the primary query/gallery set and
checkpoint selection metric. `--eval-datasets` adds cross-domain evaluations;
unavailable extra datasets are logged and skipped. Standard benchmark
query/gallery sets are often official test data. Use a separate development
split when selecting recipes, then report a fixed candidate on the held-out
benchmark.

## Training recipes

Recipes live in `boxmot/reid/training/configs/recipes`. The generated argument
reference lists every available `--recipe`; these illustrate the main choices:

| Recipe | Training inputs beyond RGB crops | Inference behavior |
| --- | --- | --- |
| `csl_tinyvit_11m` | None | 11M V20 hierarchy, 1536-D descriptor |
| `csl_tinyvit_11m_v20_pose_teacher` | Pose/person-mask metadata | Same 1536-D RGB descriptor; teacher removed |
| `csl_tinyvit_7m_v20` | Pose/person-mask metadata | 1152-D RGB descriptor; teacher removed |
| `csl_tinyvit_23m_v8` | Pose/person-mask metadata | 1920-D RGB descriptor; teacher removed |
| `mobilenetv4_conv_medium_v20`, `mobilenetv4_hybrid_medium_v20` | Pose/person-mask metadata | 1152-D RGB descriptor |
| `hi_afa_market1501` | None | 8192-D descriptor from unique normalized streams |
| `csl_tinyvit_7m_multilevel_suppression` | The V20 pose metadata | Class-CAM suppression is training-only |
| `csl_tinyvit_7m_hpgrd` | Pose metadata, person/background masks, offline teacher cache | Unchanged V20 RGB descriptor |

For the pose-teacher recipe, prepare metadata as described below, then run:

```bash
boxmot train-reid \
  --recipe csl_tinyvit_11m_v20_pose_teacher \
  --data-dir /data/Market-1501-v15.09.15 \
  --anatomical-metadata-dir /data/Market-1501-pav-metadata \
  --device cuda:0 --project runs/reid_train --name tinyvit_pose
```

The historical Hi-AFA paper scores and BoxMOT recipe assumptions are explained
in the [model reference](../config/reid-models.md). For offline teacher
extraction and cache validation, use the [HP-GRD guide](../guides/csl-tinyvit-7m-hpgrd.md).
The commands on this page use the packaged CLI and source modules; separate
research shell launchers are not required.

## Identity-preserving background mosaic

Background mosaic retains the anchor person and nearby bags, replacing only
the background with donor tiles. The anchor identity remains the training
label; query/gallery images are not augmented.

From a source checkout with the `yolo` extra, generate masks:

```bash
uv run --no-sync python -m tools.create_market1501_person_masks \
  --source datasets/reid/Market-1501-v15.09.15 \
  --output Market-1501-mosaic-highconf \
  --model yolo26x-seg.pt --device cuda:0 \
  --batch-size 16 --conf 0.50 --masks-only
```

The tool writes `Market-1501-mosaic-highconf-masks/primary/` and
`all_people/` masks. Use that generated root:

```bash
boxmot train-reid \
  --recipe csl_tinyvit_11m \
  --data-dir datasets/reid/Market-1501-v15.09.15 \
  --background-mosaic \
  --background-mosaic-mask-dir Market-1501-mosaic-highconf-masks \
  --background-mosaic-probability 0.30 --device cuda:0
```

Missing or invalid foreground masks fall back to the original anchor image.
Schedule and accepted foreground ratios are configurable; recipe values can
override the global defaults.

## Cross-camera same-ID part mosaic

`--same-id-part-mosaic` replaces body-aligned rectangles using independently
augmented observations of the same identity in the P/K batch. It prefers
cross-camera donors, preserves the identity label, and needs no segmentation
metadata.

```bash
boxmot train-reid \
  --recipe csl_tinyvit_11m \
  --data-dir datasets/reid/Market-1501-v15.09.15 \
  --same-id-part-mosaic --same-id-part-mosaic-probability 0.35 \
  --same-id-part-mosaic-max-regions 2 \
  --same-id-part-mosaic-min-unaltered 0.5 --device cuda:0
```

The camera-aware sampler can improve donor availability. Independent Random
Erasing runs after the composite when enabled.

## Pose-aligned view mosaic

PAV-Mosaic uses pose keypoints and foreground masks to align semantic parts
between same-identity observations. Generate its training metadata from the
source checkout:

```bash
uv run --no-sync python -m tools.create_market1501_pav_metadata \
  --source datasets/reid/Market-1501-v15.09.15 \
  --output Market-1501-pav-metadata \
  --pose-model yolo26x-pose.pt --seg-model yolo26x-seg.pt \
  --device cuda:0 --batch-size 16 --pose-conf 0.25 --seg-conf 0.50
```

```bash
boxmot train-reid \
  --recipe csl_tinyvit_11m \
  --data-dir datasets/reid/Market-1501-v15.09.15 \
  --pav-mosaic --pav-metadata-dir Market-1501-pav-metadata \
  --pav-mosaic-probability 0.25 --pav-mosaic-max-parts 3 \
  --pav-mosaic-min-unaltered 0.5 --device cuda:0
```

`--pav-consistency-weight` enables clean-versus-mosaic descriptor consistency,
requiring an extra clean forward for successfully augmented pairs. Keep the
metadata aligned with the actual training images.

## Privileged anatomical supervision

`--anatomical-auxiliary` supervises six ordered RGB roles: head, torso,
left/right arms, and left/right legs. The default
`deterministic_scale_aware_geometry` target uses pose/mask routing. The
`learned_pose_concat_ema` target adds a training-only pose teacher, as selected
by the bundled V20 pose recipes.

Metadata paths are explicit. `--anatomical-person-mask-dir` can supply external
foreground masks that take priority over metadata references. Startup checks
annotation coverage and file availability. Transformations keep image and
anatomical targets aligned, including left/right role swaps after horizontal
flips. Resume provenance includes annotation content.

`--anatomical-multiscale` adds local and fine supervision; their scale weights
must sum to one. Student ramp and decay options control when this signal is
active. Descriptor and branch-relation distillation are separate objectives;
use a recipe as the starting point because target/head combinations have
compatibility requirements.

Most privileged branches are removed for inference. Two explicit treatments
change this contract: `privileged_mask_pose_attention` retains its RGB gating
adapter, and `--anatomical-deployment` appends RGB-predicted semantic parts to
the descriptor. Neither requires pose or masks at inference. Descriptor
width and compute therefore depend on the selected treatment, not only the
backbone name.

## Modular CSL-TinyViT ablations

The resolved ablation plan records architecture, retrieval head, augmentation,
privileged supervision, and auxiliary objectives in `model.ablation` inside
`hparams.json` and checkpoint metadata. Preserve that configuration with
reported results.

Options such as `--head-type`, `--width-first-hierarchy`,
`--identity-registers`, and `--mcpt-mode` change the deployed model; training-only
objectives such as multilevel suppression and HP-GRD do not. Treat each
combination as its own experiment rather than transferring results between
heads or widths.

A source checkout provides an optional model/forward preflight. It resolves the
training arguments, constructs the selected model, and reports its descriptor
width and fingerprints:

```bash
uv run --no-sync python -m boxmot.engine.commands.reid.ablation validate -- \
  --recipe csl_tinyvit_11m \
  --data-dir /data/Market-1501-v15.09.15 --device cpu
```

This performs model construction and a forward pass; it is not a complete
training or accuracy test. Reusable registries live in
`boxmot.reid.backbones.head_registry`, `boxmot.reid.backbones.option_registry`,
and `boxmot.reid.training.ablation`.

## Related pages

- [Evaluate ReID](eval-reid.md)
- [Compare ReID](compare-reid.md)
- [Export](export.md)
- [ReID Profiles](../config/reid.md)

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.reid.train
    :command: train_reid
    :depth: 0
    :style: table
    :prog_name: boxmot train-reid
