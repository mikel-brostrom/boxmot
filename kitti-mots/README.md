# KITTI MOTS sensor dataset

Sequence inputs and saved detector predictions for BoxMOT EagerMOT evaluation and tuning.
From the repository root:

```bash
boxmot eval --dataset ./kitti-mots --tracker eagermot --split val --project runs/kitti-multimodal
boxmot tune --dataset ./kitti-mots --tracker eagermot --n-trials 50
```

Run one sequence and one trial to check ingestion:

```bash
boxmot tune --dataset ./kitti-mots --tracker eagermot --sequence 0002 --n-trials 1
```

Calibrate the 3D Kalman noise separately for cars and pedestrians on the training
split, then tune with those fitted values fixed:

```bash
boxmot tune --dataset ./kitti-mots --tracker eagermot --split train \
  --calibrate-kf --cache-inputs --n-trials 50
```

The `ground_truth_3d` declaration reads KITTI tracking labels with frame numbers
and persistent track IDs from `training/label_02`. Camera calibration and ego
poses come from the same per-sequence inputs used for tracking.

## Layout

```text
kitti-mots/
  dataset.yaml
  training/
    label_02/0002.txt
  sequences/
    training/0002/
      images/000000.png
      ground_truth/000000.png
      calibration.txt
      poses.npy
    testing/0000/
      images/000000.png
      calibration.txt
  predictions/
    trackrcnn/
      training/0002.txt
    pointgnn-car-t2/
      training/0002/000000.txt
    pointgnn-car-t3/
      training/0002/000000.txt
      testing/0000/000000.txt
    pointgnn-pedestrian/
      training/0002/000000.txt
      testing/0000/000000.txt
```

[dataset.yaml](dataset.yaml) uses the shared dataset schema to define classes,
frame rate, modality formats and paths, and the KITTI MOTS train/validation
partitions. Each of the 21 annotated sequences is stored once
under `sequences/training`, even when it belongs to several selections.

The same YAML selects predictions. Validation overrides `detections_3d` to use
the nine-sequence PointGNN car T2 set; `train` and `fulltrain` use car T3.
All splits use the shared pedestrian and TrackR-CNN predictions. To change a
selection, edit the modality's paths or add a complete split-specific modality
override. Separate replay and prediction manifests are no longer used.

With `storage.root: .`, all paths resolve relative to `dataset.yaml`. This folder is
self-contained and can be moved or copied to another machine. Images and ground
truth are independent macOS copy-on-write copies of the original downloads;
there are no links back to `Downloads`.

## Data conventions

- Images and instance ground truth use matching zero-based six-digit PNG names.
  Images define the timeline, including frames without predictions.
- Ground-truth PNGs retain KITTI MOTS instance labels. The configured encoding is
  `class_id * 1000 + instance_id`, with background 0 and ignore label 10000.
  Classes are car (1) and pedestrian (2).
- 3D tracking ground truth uses the original 17-field KITTI sequence labels.
  Calibration retains `Car` and `Pedestrian`; the YAML explicitly excludes the
  remaining classes, including the `Person` labels in sequences 0013 and 0019.
- `calibration.txt` retains the KITTI `P2` camera projection. `poses.npy` stores
  one absolute camera-to-world 4×4 transform per image frame.
- TrackR-CNN files retain their original mask/embedding rows. PointGNN directories
  retain their original per-frame KITTI detection rows. The YAML explicitly
  excludes Cyclist rows and maps PointGNN scores using `score_transform: odds`.
- The 29 testing sequences and available testing predictions are preserved.
  Testing lacks instance ground truth, poses, and TrackR-CNN observations here,
  so it is not a tuning split.

Detector checkpoint training provenance has not been independently verified.
Changing a prediction selection does not establish that its checkpoint was
trained independently of the selected evaluation sequences.

The original source directories are retained here for provenance:

| Input | Source directory |
| --- | --- |
| TrackR-CNN | `trackrcnn_detections` |
| PointGNN car T2 | `results_tracking_car_auto_t2_train` |
| PointGNN car T3 | `results_tracking_car_auto_t3_trainval` |
| PointGNN pedestrian/cyclist | `results_tracking_ped_cyl_auto_trainval` |

## Outputs

Evaluation writes predicted tracks under `mots/`, `metrics.json`, `metrics.csv`,
and resolved input paths in `run.json`, beneath `runs/kitti-multimodal/val` for
the example command. Scoring covers car and pedestrian mask tracking. 3D box
metrics are not implemented.

Tuning writes a new directory under `runs/eagermot-tune/<split>` with `best.yaml`,
an Optuna study, per-trial metrics, and resolved input paths in `run.json`.
With `--calibrate-kf`, `kf-tuning/calibrated.yaml` and
`kf-tuning/calibration.json` contain the fitted class profiles and their evidence.
Replay a selected profile with:

```bash
boxmot eval --tracker eagermot --dataset ./kitti-mots --sequence 0002 \
  --class-config runs/eagermot-tune/val/best.yaml
```

The local `.gitignore` includes `dataset.yaml` and this README while excluding images,
ground truth, poses, calibration files, and prediction payloads from version control.
