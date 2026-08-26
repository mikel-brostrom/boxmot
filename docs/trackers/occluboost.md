# OccluBoost

OccluBoost is an occlusion-aware hybrid tracker built on top of BoostTrack. It
keeps BoostTrack's multi-cue association and confidence boosting, then adds
tentative-track confirmation, ReID recovery, a guarded low-confidence second
pass, duplicate suppression, and an **Abnormal Motion Suppression (AMS)**
Kalman update. The Python implementation can also enable online global
trajectory association (GTA) for longer appearance-based recovery.

## What's layered on top of BoostTrack

- **AMS Kalman update.** Every matched Kalman update (first pass, ReID recovery, low-conf second pass) is routed through `_ams_update`, which scales the Kalman gain on the mean update by `alpha ∈ [ams_alpha0, 1]` when an abnormal-motion event is detected. The covariance still uses the standard update; only the mean correction is suppressed.
    - **Trigger.** A per-track ring buffer of length `ams_buffer_size` tracks `[cx, cy, w, h]`. We compute the relative speed spike of the centre and aspect against the buffer mean; if either exceeds `ams_threshold`, the speed gate fires.
    - **Shrink gate (key addition over the OccluTrack paper).** Suppression only kicks in when the new detection is also physically smaller than the running mean: `cur_area < ams_shrink_ratio * mean_area`. This keeps pure speed spikes from being treated as partial occlusion.
    - **OBB safety.** OBB tracks bypass AMS (`alpha=1.0`) — the suppression model is defined for AABB motion only.
- **BotSort-style track confirmation** (`tentative -> activated`). New tracks born from medium-confidence detections must accumulate `confirm_hits` consecutive matches before being emitted; detections above `instant_confirm_thresh` skip the wait. Tentative tracks expire after `tentative_max_age` frames, slashing ghost IDs from one-frame flickers.
- **ReID-only recovery pass.** Unmatched high-confidence detections are re-attached to recently lost tracks when cosine appearance similarity exceeds `recovery_appearance_thresh` and a loose IoU sanity gate (`recovery_iou_thresh`) is satisfied. Recovered embeddings are EMA-blended with `feat_alpha`.
- **Safe appearance-gated second pass.** Low-confidence detections (`track_low_thresh ≤ conf < det_thresh`) can re-attach **only** to confirmed tracks (`is_activated=True`) under strict IoU + appearance gates. This lifts MOTA without the ID switches an unrestricted ByteTrack-style second pass introduces.
- **Duplicate suppression.** `duplicate_iou_thresh` controls removal of the younger of two near-identical emitted tracks.
- **Optional online GTA.** When `gta_enabled` is set, appearance-only recovery can reconnect eligible live tracks, resurrect recently removed tracks from a graveyard, and optionally interpolate and smooth recovered gaps. The built-in tracker config leaves GTA disabled.

## What BoxMOT Needs For OccluBoost

- A detector and a ReID model (the recovery pass and second-pass appearance gate both rely on embeddings).
- AABB or OBB detections. OBB inputs use oriented IoU, OBB-aware confidence
  boosting, optional ReID recovery and second-pass matching, and the 9-column
  output schema `[cx, cy, w, h, angle, id, conf, cls, det_ind]`.
- Best for crowded / partial-occlusion scenes where identity preservation matters.

## Native C++ Backend

BoxMOT ships a native C++17 OccluBoost implementation under
`boxmot/native/cpp/trackers/occluboost/`. It implements the core association,
confirmation, recovery, second-pass, duplicate-suppression, and AMS paths and
supports:

- cached replay for `eval` and `tune`
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detections for live tracking and cached replay
- ReID inference through the shared native `OnnxReIdModel`, used for the first-pass association, the ReID-only recovery pass, and the appearance-gated low-confidence second pass
- automatic `.pt -> .onnx` export for native cpp inference when you pass PyTorch ReID weights

Online GTA and adaptive-Kalman controls are currently Python-only; selecting
the C++ backend does not enable those two extensions.

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker occluboost --tracker-backend cpp
boxmot track --tracker occluboost --tracker-backend cpp --reid models/lmbn_n_duke.pt --source 0
```

When `--tracker-backend cpp` is set, cached replay requests the native C++ ReID
producer and stores its embeddings under `embs/cpp/`. Python-generated
embeddings use a separate `embs/python/` bucket; native initialization and model
errors are surfaced rather than silently changing producer. If the native ReID
module cannot be imported, backend resolution selects the Python producer first
and therefore writes to `embs/python/`. These buckets
describe how embeddings were computed, not which tracker algorithm consumes
them. See [Native C++ Integration](../native/index.md#embedding-cache-layout)
for the full layout and the runtime knobs (`BOXMOT_REID_BACKEND`,
`BOXMOT_REID_DEVICE`).

## Tuning notes

The canonical defaults and tuning metadata live together in
`boxmot/configs/trackers/occluboost.yaml`; consult that file instead of copying
numeric values into a custom config. The main parameter groups are:

- `ams_enabled`, `ams_alpha0`, `ams_threshold`, `ams_shrink_ratio`, and
  `ams_buffer_size` for AABB abnormal-motion suppression. Lower `ams_alpha0`
  suppresses the mean update more strongly when both the motion and shrink
  gates fire.
- `confirm_hits`, `instant_confirm_thresh`, and `tentative_max_age` for the
  tentative pool. Fewer confirmation hits emit tracks sooner but admit more
  short-lived false positives.
- `recovery_*`, `feat_alpha`, and `with_reid` for appearance recovery.
- `use_second_pass`, `second_*`, and `track_low_thresh` for guarded
  low-confidence association.
- `gta_*` for the optional Python-only global trajectory association path.
- `obb_*` for thresholds and lifetimes that intentionally differ in OBB mode.
- `new_track_thresh` and `max_age` for new-track creation and gap tolerance.

### Adaptive Kalman Filter (`adaptive_kf`)

When `adaptive_kf: true` is set in the tracker config, the process noise covariance **Q** is estimated online from innovation statistics (Mehra 1970) rather than kept constant. A sliding window (30 frames, warmup 15) accumulates the outer products of the Kalman innovations, and once warmed up the estimated Q is blended (α = 0.7) with the default static Q.

**When to use it:**

- Deploying to a new domain where you have no ground truth to run `--tune-kf`.
- Scenes where camera motion compensation (CMC) may fail intermittently (low-texture, rain, night).
- Camera dynamics that vary significantly within a single sequence (e.g., drone footage alternating hover and fast sweep).

**When NOT to use it:**

- You already have a tuned static Q from `boxmot eval --tune-kf` on representative data — the static solution is cheaper and deterministic.
- Very short tracks (< 15 frames) dominate; the estimator never exits warmup so it adds overhead with no benefit.

Enable it through the Python facade:

```python
from boxmot import BoxMOT

model = BoxMOT(
    tracker="occluboost",
    tracker_kwargs={"adaptive_kf": True},
)
model.track(source="video.mp4")
```

Or set it in a custom tracker config YAML:

```yaml
adaptive_kf: true
```

Use `boxmot eval --tune-kf` when you want a calibrated static Kalman model.
Tracker tuning can also explore `adaptive_kf` because it is declared as a
choice in the built-in search space.

::: boxmot.trackers.bbox.occluboost.OccluBoost
