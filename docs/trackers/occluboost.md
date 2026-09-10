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

- A detector and appearance embeddings (the recovery pass and second-pass
  appearance gate rely on them). Both the Python implementation and native
  adapter can extract missing embeddings from a supplied `Frame` or consume
  embeddings already attached to `Detections`.
- AABB or OBB detections. OBB inputs use oriented IoU, OBB-aware confidence
  boosting, optional ReID recovery and second-pass matching, and the 9-column
  output schema `[cx, cy, w, h, angle, id, conf, cls, det_ind]`.
- Best for crowded / partial-occlusion scenes where identity preservation matters.

## Native C++ Backend

BoxMOT ships a native C++17 OccluBoost implementation under
`boxmot/native/cpp/trackers/occluboost/`. It implements the core association,
confirmation, recovery, second-pass, duplicate-suppression, and AMS paths and
supports:

- cached `eval` and `tune` streamed through the live typed API
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detections for live tracking and cached evaluation
- typed generated or precomputed embeddings for association and recovery through the v2 update ABI
- model-free C++ tracker code; optional ReID inference is owned by its Python adapter

Online GTA and adaptive-Kalman controls are currently Python-only; selecting
the C++ backend does not enable those two extensions.

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker occluboost --tracker-backend cpp
boxmot track --tracker occluboost --tracker-backend cpp --reid models/lmbn_n_duke.pt --source 0
```

Native and Python OccluBoost both accept embeddings already attached to
canonical `Detections`. When `use_embeddings=True` and a non-empty batch has no
embeddings, the high-level tracker lazily initializes its configured encoder or
backend and extracts embeddings from the supplied `Frame`. A native adapter
then passes that typed feature buffer to its model-free C++ library.
Configuration may come from a complete `ReIDEncoderSpec`, an injected
`reid_model`, or `reid_weights`, `device`, `half`, and `reid_preprocess`. If no
weights or spec are supplied, the default ReID model is used.
Attached embeddings bypass inference, and empty batches do not initialize the
model. See [Live embeddings in ReID-enabled
trackers](../python/index.md#live-embeddings-in-reid-enabled-trackers).
Materialized builds record the encoder fingerprint and publish embeddings as
keyed Parquet rows.
See [Native C++ Integration](../native/index.md#capabilities-and-requirements).

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
- `recovery_*`, `feat_alpha`, and `use_embeddings` for appearance recovery.
- `use_second_pass`, `second_*`, and `track_low_thresh` for guarded
  low-confidence association.
- `gta_*` for the optional Python-only global trajectory association path.
- `obb_*` for thresholds and lifetimes that intentionally differ in OBB mode.
- `new_track_thresh` and `max_age` for new-track creation and gap tolerance.

### Adaptive Kalman Filter (`adaptive_kf`)

The Python implementation supports experimental online process-noise estimation
with `adaptive_kf=True` (default: `False`). It uses a window of up to 30 Kalman
innovations per track, starts adapting after 15 measurement corrections, and
blends the estimate with baseline noise (70% adaptive, 30% baseline).
Initialization and prediction-only updates do not count toward warmup.
Measurement noise remains configured separately.

Consider adaptation for long tracks whose motion predictability changes, then
compare against validated fixed noise settings. Short tracks may never leave
warmup; detector, association, and camera-compensation errors can distort the
estimate. `variable_dt=True` independently handles irregular capture intervals
and can be combined with adaptation. See
[choosing Kalman timing and adaptation](../modes/track.md#choose-kalman-timing-and-adaptation)
for scenarios and CLI examples.

Enable it through the structured factory:

```python
from boxmot import create_tracker
from boxmot.trackers import TrackerSpec

tracker = create_tracker(
    TrackerSpec(
        name="occluboost",
        options=(("adaptive_kf", True),),
    )
)
```

Or set it in a custom tracker config YAML:

```yaml
adaptive_kf: true
```

Use a custom tracker configuration when you have calibrated static Kalman
parameters. `adaptive_kf` is a runtime setting and stays fixed during tracker
tuning, along with the calibrated covariance scales.

::: boxmot.OccluBoost
