# BoostTrack

[Paper: BoostTrack++: Using Tracklet Information to Detect More Objects in Multiple Object Tracking](https://arxiv.org/abs/2408.13003)

BoostTrack++ focuses on a neglected part of MOT pipelines: deciding which detections are worth trusting in the first place. The paper extends BoostTrack by using tracklet history to build a richer similarity score, then boosts low-confidence detections when past evidence suggests they are real objects. In practice, that improves recall and identity stability without giving up the online tracking-by-detection setup.

## What BoxMOT Needs For BoostTrack

- A detector and, by default, a ReID model for the full configuration.
- AABB detections only in BoxMOT.
- Best when low-confidence true positives are a recurring problem and you want stronger association scoring than plain IoU or Mahalanobis distance.

## Tuning notes

### Adaptive Kalman Filter (`adaptive_kf`)

When enabled, the process noise covariance **Q** is estimated online from innovation statistics (Mehra 1970) rather than kept constant. A sliding window (30 frames, warmup 15) accumulates the Kalman innovations, and once warmed up the estimated Q is blended (α = 0.7) with the default static Q.

**When to use it:**

- Deploying to a new domain where you have no ground truth to run `--tune-kf`.
- Scenes where camera motion compensation (CMC) may fail intermittently (low-texture, rain, night).
- Camera dynamics that vary significantly within a single sequence (e.g., drone footage alternating hover and fast sweep).

**When NOT to use it:**

- You already have a tuned static Q from `boxmot eval --tune-kf` on representative data — the static solution is cheaper and deterministic.
- Very short tracks (< 15 frames) dominate; the estimator never exits warmup so it adds overhead with no benefit.

Enable it from the CLI:

```bash
boxmot track --tracker boosttrack --adaptive-kf
boxmot eval  --tracker boosttrack --adaptive-kf
```

Or in the tracker config YAML:

```yaml
adaptive_kf: true
```

::: boxmot.trackers.bbox.boosttrack.BoostTrack
