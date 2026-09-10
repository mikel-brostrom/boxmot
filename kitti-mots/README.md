# KITTI MOTS sensor dataset

Sequence inputs and saved detector predictions for BoxMOT EagerMOT tuning.
From the repository root:

```bash
boxmot tune --dataset ./kitti-mots --tracker eagermot --n-trials 50
```

Run one sequence and one trial to check ingestion:

```bash
boxmot tune --dataset ./kitti-mots --tracker eagermot --sequence 0002 --n-trials 1
```

## Layout

```text
kitti-mots/
  dataset.yaml
  replay.yaml
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
      manifest.yaml
      training/0002.txt
    pointgnn-car-t2/
      manifest.yaml
      training/0002/000000.txt
    pointgnn-car-t3/
      manifest.yaml
      training/0002/000000.txt
      testing/0000/000000.txt
    pointgnn-pedestrian/
      manifest.yaml
      training/0002/000000.txt
      testing/0000/000000.txt
```

[dataset.yaml](dataset.yaml) defines classes, sequence paths, and the KITTI MOTS
train/validation partitions. Each of the 21 annotated sequences is stored once
under `sequences/training`, even when it belongs to several selections.

[replay.yaml](replay.yaml) selects the image, car, and pedestrian prediction
manifests for each split. Validation uses the nine-sequence PointGNN car T2 set;
`train` and `fulltrain` use car T3. To change predictions, edit the manifest
selection for that split. Each prediction manifest records its file format,
class IDs, available sequences, and source directory.

All paths resolve relative to the manifest that declares them. This folder is
self-contained and can be moved or copied to another machine. Images and ground
truth are independent macOS copy-on-write copies of the original downloads;
there are no links back to `Downloads`.

## Data conventions

- Images and instance ground truth use matching zero-based six-digit PNG names.
  Images define the timeline, including frames without predictions.
- Ground-truth PNGs retain KITTI MOTS instance labels. Classes are car (1) and
  pedestrian (2).
- `calibration.txt` retains the KITTI `P2` camera projection. `poses.npy` stores
  one absolute camera-to-world 4×4 transform per image frame.
- TrackR-CNN files retain their original mask/embedding rows. PointGNN directories
  retain their original per-frame KITTI detection rows; BoxMOT excludes cyclists.
- The 29 testing sequences and available testing predictions are preserved.
  Testing lacks instance ground truth, poses, and TrackR-CNN observations here,
  so it is not a tuning split.

Detector checkpoint training provenance has not been independently verified.
Changing a prediction selection does not establish that its checkpoint was
trained independently of the selected evaluation sequences.

## Outputs

Tuning writes a new directory under `runs/eagermot-tune/<split>` with `best.yaml`,
an Optuna study, per-trial metrics, and resolved input paths in `run.json`.
Replay a selected profile with:

```bash
boxmot eval --tracker eagermot --dataset ./kitti-mots --sequence 0002 \
  --class-config runs/eagermot-tune/val/best.yaml
```

The local `.gitignore` includes manifests and this README while excluding images,
ground truth, poses, calibration files, and prediction payloads from version control.
