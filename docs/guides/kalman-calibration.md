# Kalman calibration

Fit Kalman noise from cached detections and ground truth, select a filter on
held-out predictions, then tune tracker parameters against tracking metrics.
Selection can retain original defaults when fitted noise predicts less well.

## Choose a workflow

| Command | Behavior |
| --- | --- |
| `eval --calibrate-kf` | Fit the configured filter and evaluate once. |
| `tune --calibrate-kf` | Fit the configured filter, then tune tracker parameters. |
| `tune --select-kf` | Compare compatible filter types and noise profiles, fix the winner and CMC off, then tune tracker parameters. |
| `tune --select-kf --select-cmc` | Jointly select filter type, noise profile, and CMC off or automatic SOF before tuning tracker parameters. |

Selection includes fitting and does not require `--calibrate-kf`. It supports
Python ByteTrack, BoT-SORT, OC-SORT, DeepOC-SORT, HybridSORT, StrongSORT,
BoostTrack, and OccluBoost with AABB or OBB data. Native backends and trackers
without Kalman filters are unsupported. EagerMOT uses its separate
[3D calibration workflow](../trackers/eagermot.md#calibrate-3d-kalman-noise).

```bash
boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort --select-kf --select-cmc --n-trials 200
```

This prepares or reuses the experiment's perception build. Add `--build BUILD_ID`
to select an existing compatible build, or `--tracker-config path/to/best.yaml`
to supply starting tracker settings. Source images must remain available for
camera analysis.

## Select a filter and noise profile

`--select-kf` compares compatible XYAH, XYWH, XYSR, XYHR, and XYSCR filters,
using four noise recipes for each:

| Profile | Process and initial covariance | Measurement covariance |
| --- | --- | --- |
| `default` | Original | Original |
| `motion` | Fitted | Original |
| `measurement` | Original | Fitted |
| `calibrated` | Fitted | Fitted |

Up to three folds hold out complete sequences. A single sequence instead holds
out complete object trajectories. Selection requires at least two usable groups;
training folds need sufficient matched detections to fit noise. Prior fitted
noise is cleared before fitting the training folds.

Ground-truth identities provide correspondence and detections update the
filters. Selection scores forecast IoU on held-out trajectories, averaging
within trajectories and sequences so long sequences do not dominate. Plain
KF selection uses one-, three-, and five-frame horizons. Joint CMC selection
ranks one-frame forecasts and reports longer gaps separately.

The winning fitted recipe is refitted on all selected data. A default winner
retains original noise. The chosen type and noise stay fixed during tracker
tuning. Add repeated `--tune-kf SCALE` options to explicitly refine selected
covariance scales from 0.25 to 4 times their selected values; see
[Kalman noise and timing](../modes/tune.md#kalman-noise-and-timing).

Internal held-out prediction scores do not measure identity association or
HOTA. Evaluate the final `best.yaml` on separate sequences before claiming a
tracking improvement.

## Select camera compensation

`--select-cmc` requires `--select-kf`. Joint selection compares CMC off and
automatic sparse optical flow (SOF) for every eligible filter and noise recipe.
It fits each candidate under its own camera behavior and fixes the winning
camera strategy throughout tracker tuning.

Automatic CMC enables SOF where accepted camera transforms show sustained
motion. Setup first checks `<build>/cmc/<analysis-key>/` for compatible evidence
from an earlier evaluation or tuning run. On a miss, it analyzes source images,
processing independent sequences in parallel up to `--sequence-workers` while
retaining frame order. It stores `analysis.json` and `warps.npz` for later runs
without changing the canonical build or rerunning perception.

Reuse validates source files, frame identities, tracker masking, estimator
settings, and class and sequence selections. Source images must remain
available, although reuse skips decoding and sparse optical flow. Trials share
the validated transforms and skip image decoding when CMC is the only component
needing pixels. This storage also applies to joint `--select-cmc` selection.
Selection without `--select-cmc` keeps CMC off. Other workflows keep their
usual tracker settings.

A saved automatic strategy prepares or reuses evidence for each evaluation
run's current inputs. New inputs receive their own sequence decisions. See
[CMC by sequence](../modes/eval.md#cmc-by-sequence).

## Outputs and resume

Selection writes the following under the tuning run:

| Artifact | Contents |
| --- | --- |
| `kf-tuning/filters.json` | Filter-profile bank and artifact signatures. |
| `kf-tuning/<type>/<profile>.yaml` | Reusable selected noise recipes. |
| `kf-tuning/selection.json` | Fold scores, winning filter/profile, and selected configuration path. |
| `kf-tuning/cmc/analysis.json`, `kf-tuning/cmc/warps.npz` | Run copies of joint selection's stored camera evidence. |
| `kf-tuning/cmc-policy.yaml` | Winning decisions for the fitting sequences. |
| `best.yaml` | Winning tracker settings, selected KF, and camera strategy. |

Camera analysis uses the `boxmot.cmc-analysis/v2` schema. Its archive stores
both policy transforms and transforms matching the tracker's runtime masks,
with source and frame provenance checked before reuse. Older camera evidence
requires fresh selection before resuming a joint CMC/KF tuning run.

Reuse the final result with:

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort --tracker-config path/to/best.yaml
```

Use the actual path reported by tuning. This example replays the fitting split;
select a held-out experiment or split for final evaluation. Keep detector,
geometry, class selection, and timing consistent when reusing a profile.

Resume with the same data/build and `--resume-tune TUNING_DIRECTORY`, omitting
`--calibrate-kf`, `--select-kf`, and `--select-cmc`. The saved choice and search
space are restored. Repeat any original `--tune-kf` fields. Fresh selection
cannot replace the saved choice in a resumed search.

Fixed-step prediction remains the default. `--variable-dt` fits and predicts
using capture timestamps; `--fps` selects frames without changing that timing
mode. See [Kalman calibration](../modes/eval.md#kalman-calibration) for fitting
and timing details.
