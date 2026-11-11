# Supported datasets

BoxMOT includes tracking dataset profiles for axis-aligned boxes, oriented boxes,
and instance masks, plus readers for saved 2D/3D sensor predictions. Choose a
profile below, then use an [experiment](../config/experiments.md) to select its
split and perception models. Dataset profiles live in `boxmot/configs/datasets`.

## Tracking datasets

The splits below are the names exposed by the bundled profiles. AABB means
axis-aligned boxes; OBB means oriented boxes.

| Dataset / profile | Geometry and classes | Configured splits | Data setup |
| --- | --- | --- | --- |
| MOT17 · `mot17` | AABB; pedestrians | `train`, `val`, `ablation` (default), `test` | Automatic download of the selected split |
| MOT20 · `mot20` | AABB; pedestrians | `train`, `ablation` (default), `test` | Automatic download of the selected split |
| DanceTrack · `dancetrack` | AABB; pedestrians | `val` (default) | Prepare the validation sequences locally under `DanceTrack/val` |
| SportsMOT · `sportsmot` | AABB; players | `train`, `val` (default), `test` | Automatic download of the selected split |
| VisDrone · `visdrone` | AABB; 10 classes | `val` (default) | Extract the configured archive; this profile maps `val` to `VisDrone2019-MOT-test-dev/sequences` |
| MMOT · `mmot` | OBB; 8 classes, multispectral frames | `train` (default), `test` | Automatic download of the selected split; frames in `<split>/npy`, annotations in `<split>/mot` |
| KITTI MOTS / KITTI tracking · `kitti-mots` | Image boxes, masks, and 3D boxes; cars and pedestrians | `train`, `val` (default), `fulltrain`, `test` | Supply images, annotations, and the saved predictions needed by the selected workflow |

MOT17, MOT20, SportsMOT, and KITTI `test` splits have no local ground truth in
these profiles. MMOT's `test` split does include annotations. KITTI uses 12
training and nine validation sequences; `fulltrain` selects all training sequences.

Automatic acquisition applies only to configured Hugging Face `per_split`
resources when the selected split is missing. Existing populated splits are
reused. Archive resources require manual download and extraction; the profile's
`resources` field records the download location. DanceTrack requires validation
data even though its current archive entry points to `test1.zip`.

Built-in profile paths resolve beneath `./datasets/mot` by default. Use
`--data-root` to choose another base directory. For local layouts, class IDs,
split definitions, and supported annotation encodings, see the
[dataset configuration reference](../config/datasets.md).

`--data-root` is the parent of the configured `storage.root`. For example,
MOT17 uses `<data-root>/MOT17`, MMOT uses `<data-root>/data`, and the KITTI
profile uses `<data-root>` directly because its root is `.`. The split names
are configuration selections: MOT17's `val` and `ablation` are the bundled
partitions, and VisDrone's `val` points to the test-dev payload rather than
the official validation directory.

MMOT's raw `.npy` frames must be uint8 arrays in height-width-channel order.
For its eight-band images, BoxMOT selects zero-based RGB bands `[4, 2, 1]`,
matching the bundled `yolo11l-mmot-obb` checkpoint; it does not run an
eight-channel detector. See [detector profiles](../config/detectors.md).

## Run a benchmark

From a [source checkout with the `yolo` extra](../getting-started/pytorch.md#source-checkout),
materialize MOT17's ablation split, then evaluate a tracker against the resulting
detections and appearance embeddings:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID --tracker botsort
```

Replace `BUILD_ID` with the ID printed by `materialize`, or use the full build
path. The first command downloads missing split data and model weights. Reuse
that build when [comparing trackers](../guides/comparison.md). See
[Materialize](../modes/materialize.md) and [Evaluation](../modes/eval.md) for
storage and output options.

## Small development datasets

| Profile | Geometry | Split | Setup |
| --- | --- | --- | --- |
| `mot17-mini` | AABB | `train` | Included in a repository checkout at `assets/MOT17-mini`; use `--data-root assets` |
| `mmot-mini` | OBB | `train` | Manually extract the configured `MMOT-OBB.zip` archive to match `MMOT-mini/train/npy` beneath your data root |

From a repository checkout, use the small MOT17 fixture for a workflow smoke test:

```bash
boxmot materialize --experiment mot17-mini/train-yolo26n-lmbn.yaml --data-root assets
boxmot eval --experiment mot17-mini/train-yolo26n-lmbn.yaml \
  --data-root assets --build BUILD_ID --tracker bytetrack
```

The mini profiles are development fixtures. Use the full benchmark profiles for
reported results.

## KITTI and custom sensor datasets

The shared KITTI profile declares images, TrackR-CNN detections and masks,
PointGNN 3D detections, calibration, ego poses, and annotations. Experiments
select the required modalities from that inventory:

| Workflow | Guide |
| --- | --- |
| Evaluate saved 2D boxes with an image tracker | [KITTI 2D tracking](../config/datasets.md#kitti-2d-tracking) |
| Evaluate boxes or instance masks from a perception build | [KITTI MOTS evaluation](../guides/evaluation.md#kitti-mots-evaluation) |
| Replay saved masks and images with MAF-HDA | [MAF-HDA saved predictions](../trackers/maf_hda.md#evaluate-trackr-cnn-detections-on-kitti-mots) |
| Evaluate or tune 2D/3D fusion, or evaluate 3D tracks | [EagerMOT](../trackers/eagermot.md) |

For your own sensor data, copy the `sensor-fusion.yaml` template to a local
`dataset.yaml` and declare the paths and encodings for each modality. Follow the
[sensor dataset setup](../config/datasets.md#bring-your-own-sensor-dataset).
The current saved-sensor evaluation and tuning workflows use EagerMOT with one
calibrated camera per sequence and car/pedestrian metrics.

## ReID datasets

ReID training and retrieval evaluation use identity-labeled crops rather than
tracking sequences. The registered dataset names are `market1501`, `mot171501`
(MOT17 crops in Market1501 format), `cuhk03`, `duke`, `msmt17`, and `veri`
(VeRi-776). `msmt17_merged` is a training variant that combines MSMT17's train,
query, and gallery sets.

The merged variant has seen MSMT17 evaluation images during training; do not
report its own query/gallery scores as held-out performance. All evaluation
commands load the complete registered dataset layout, including the training
split.

Market1501 can be downloaded automatically when training without `--data-dir`.
Provide local data for the other readers. See [ReID training](../modes/train.md#supported-datasets)
and [ReID evaluation](../modes/eval-reid.md) for the expected layouts and commands.
