# OccluBoost

OccluBoost is an occlusion-aware hybrid tracker built on top of BoostTrack. It keeps BoostTrack's multi-cue association (IoU + Mahalanobis + shape similarity, optional ReID) and DLO confidence boosting, then layers on a BotSort-inspired confirmation state, a ReID-only recovery pass, a safe low-confidence second pass, and an OccluTrack-style **Abnormal Motion Suppression (AMS)** Kalman filter that protects tracks during partial occlusion.

On the MOT17 ablation split (`yolox_x_MOT17_ablation` + `lmbn_n_duke`), OccluBoost beats BotSort on 5/6 metrics with the locked defaults: HOTA **70.47**, MOTA **78.32**, IDF1 **84.14**, IDSW **135**, AssA **74.73** (only DetA −0.27 vs BotSort).

## What's layered on top of BoostTrack

- **AMS Kalman update.** Every Kalman update (first pass, ReID recovery, low-conf second pass) is routed through `_ams_update`, which scales the Kalman gain on the mean update by `alpha ∈ [ams_alpha0, 1]` when an abnormal-motion event is detected. Covariance is left untouched so uncertainty grows naturally during the suppressed step.
    - **Trigger.** A per-track ring buffer of length `ams_buffer_size` tracks `[cx, cy, w, h]`. We compute the relative speed spike of the centre and aspect against the buffer mean; if either exceeds `ams_threshold`, the speed gate fires.
    - **Shrink gate (key addition over the OccluTrack paper).** Suppression only kicks in when the new detection is also physically smaller than the running mean: `cur_area < ams_shrink_ratio * mean_area`. Without this gate, pure speed spikes over-suppress legitimate fast motion and DetA collapses; with it, we get the IDF1/HOTA gain without losing detection accuracy.
    - **OBB safety.** OBB tracks bypass AMS (`alpha=1.0`) — the suppression model is defined for AABB motion only.
- **BotSort-style track confirmation** (`tentative -> activated`). New tracks born from medium-confidence detections must accumulate `confirm_hits` consecutive matches before being emitted; detections above `instant_confirm_thresh` skip the wait. Tentative tracks expire after `tentative_max_age` frames, slashing ghost IDs from one-frame flickers.
- **ReID-only recovery pass.** Unmatched high-confidence detections are re-attached to recently lost tracks when cosine appearance similarity exceeds `recovery_appearance_thresh` and a loose IoU sanity gate (`recovery_iou_thresh`) is satisfied. Recovered embeddings are EMA-blended with `feat_alpha`.
- **Safe appearance-gated second pass.** Low-confidence detections (`track_low_thresh ≤ conf < det_thresh`) can re-attach **only** to confirmed tracks (`is_activated=True`) under strict IoU + appearance gates. This lifts MOTA without the ID switches an unrestricted ByteTrack-style second pass introduces.
- **Duplicate suppression.** A conservative `duplicate_iou_thresh` (default 0.95) drops the younger of two near-identical emitted tracks.

## What BoxMOT Needs For OccluBoost

- A detector and a ReID model (the recovery pass and second-pass appearance gate both rely on embeddings).
- AABB or OBB detections. OBB inputs are routed through a dedicated OBB code path that uses oriented IoU for association and a 9-column output schema (`[cx, cy, w, h, angle, id, conf, cls, det_ind]`); the OBB association path replaces the AABB-specific confidence-boosting and Mahalanobis matching logic.
- Best for crowded / partial-occlusion scenes where identity preservation matters.

## Native C++ Backend

BoxMOT ships a native C++17 OccluBoost implementation under `boxmot/native/cpp/trackers/occluboost/`. It mirrors the Python tracker and shares the BoTSORT-style ReID plumbing, so it supports:

- cached replay for `eval`, `tune`, and `research`
- live `track` through `--tracker-backend cpp`
- AABB detections (OBB still goes through the Python path)
- ReID inference through the shared native `OnnxReIdModel`, used for the first-pass association, the ReID-only recovery pass, and the appearance-gated low-confidence second pass
- automatic `.pt -> .onnx` export for native cpp inference when you pass PyTorch ReID weights

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker occluboost --tracker-backend cpp \
  --detector yolox_x_MOT17_ablation.pt --reid models/lmbn_n_duke.onnx
boxmot track --tracker occluboost --tracker-backend cpp \
  --reid models/lmbn_n_duke.pt --source 0
```

When `--tracker-backend cpp` is set, embedding generation for cached replay also goes through the native C++ ReID and is written to a `__cpp`-suffixed cache bucket. See [Native C++ Integration](../native/index.md#native-c-reid) for the runtime knobs (`BOXMOT_REID_BACKEND`, `BOXMOT_REID_DEVICE`).

## Tuning notes

- **AMS knobs** (locked on the MOT17 ablation split but worth retuning per dataset):
    - `ams_alpha0` (default 0.4): how strongly to suppress the gain when both gates fire. Lower = stronger suppression. 0.3 over-protects and inflates IDSW; 0.5+ recovers IDSW but loses HOTA.
    - `ams_threshold` (default 0.5): relative speed-spike trigger. Lower fires more often.
    - `ams_shrink_ratio` (default 0.75): only suppress when the new bbox shrinks below this fraction of the buffered mean area. Disable AMS entirely with `ams_enabled: false`.
    - `ams_buffer_size` (default 30 frames).
- `confirm_hits` (default 4) and `instant_confirm_thresh` (default 0.77) control the tentative pool. Lower the threshold to emit faster (better recall, more FPs); raise `confirm_hits` to be stricter.
- `recovery_appearance_thresh` is the dominant identity safety knob: raise it (e.g. 0.7) to be conservative and protect IDF1, lower it (e.g. 0.4) to recover more occluded objects.
- `use_second_pass` is on by default and only re-attaches low-confidence detections to **confirmed** tracks above `second_pass_min_hits`. Tighten `second_iou_thresh` / `second_appearance_thresh` if you see ID switches in dense scenes; relax them to gain MOTA in clean scenes.
- `new_track_thresh` is decoupled from `det_thresh` so weakly-confident detections can update existing tracks without spawning new ones.
- Keep `max_age` comfortably above the longest expected detector gap for each class so per-class tracking survives sparse class-specific detections.

### Adaptive Kalman Filter (`adaptive_kf`)

When `adaptive_kf: true` is set in the tracker config, the process noise covariance **Q** is estimated online from innovation statistics (Mehra 1970) rather than kept constant. A sliding window (30 frames, warmup 15) accumulates the outer products of the Kalman innovations, and once warmed up the estimated Q is blended (α = 0.7) with the default static Q.

**When to use it:**

- Deploying to a new domain where you have no ground truth to run `--tune-kf`.
- Scenes where camera motion compensation (CMC) may fail intermittently (low-texture, rain, night).
- Camera dynamics that vary significantly within a single sequence (e.g., drone footage alternating hover and fast sweep).

**When NOT to use it:**

- You already have a tuned static Q from `boxmot eval --tune-kf` on representative data — the static solution is cheaper and deterministic.
- Very short tracks (< 15 frames) dominate; the estimator never exits warmup so it adds overhead with no benefit.

Enable it from the CLI:

```bash
boxmot track --tracker occluboost --adaptive-kf
boxmot eval  --tracker occluboost --adaptive-kf
```

Or in the tracker config YAML:

```yaml
adaptive_kf: true
```

The tuner will also explore it automatically since it's registered as a `choice` parameter in the search space.

::: boxmot.trackers.bbox.occluboost.OccluBoost
