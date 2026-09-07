# BoostTrack

[Paper: BoostTrack++: Using Tracklet Information to Detect More Objects in Multiple Object Tracking](https://arxiv.org/abs/2408.13003)

BoostTrack++ focuses on a neglected part of MOT pipelines: deciding which detections are worth trusting in the first place. The paper extends BoostTrack by using tracklet history to build a richer similarity score, then boosts low-confidence detections when past evidence suggests they are real objects. In practice, that improves recall and identity stability without giving up the online tracking-by-detection setup.

## What BoxMOT Needs For BoostTrack

- A detector plus appearance embeddings when `use_embeddings=True`, as in the
  default full configuration. The Python implementation can generate missing
  embeddings from a supplied `Frame` or consume embeddings already attached to
  `Detections`.
- Supports both AABB and OBB detections in BoxMOT.
- Best when low-confidence true positives are a recurring problem and you want stronger association scoring than plain IoU or Mahalanobis distance.

Direct construction accepts the shared `reid_model`, `reid_weights`, `device`,
`half`, and `reid_preprocess` options described in the
[Python API](../python/index.md#live-embeddings-in-reid-enabled-trackers).

## Tuning notes

### Adaptive Kalman Filter (`adaptive_kf`)

When enabled, the process noise covariance **Q** is estimated online from innovation statistics (Mehra 1970) rather than kept constant. A sliding window (30 frames, warmup 15) accumulates the Kalman innovations, and once warmed up the estimated Q is blended (α = 0.7) with the default static Q.

**When to use it:**

- Deploying to a new domain where you do not yet have tuned static motion parameters.
- Scenes where camera motion compensation (CMC) may fail intermittently (low-texture, rain, night).
- Camera dynamics that vary significantly within a single sequence (e.g., drone footage alternating hover and fast sweep).

**When NOT to use it:**

- You already have validated static motion parameters — the static solution is cheaper and deterministic.
- Very short tracks (< 15 frames) dominate; the estimator never exits warmup so it adds overhead with no benefit.

Enable it through the structured factory:

```python
from boxmot import create_tracker
from boxmot.trackers import TrackerSpec

tracker = create_tracker(
    TrackerSpec(
        name="boosttrack",
        options=(("adaptive_kf", True),),
    )
)
```

Or set it in a custom tracker config YAML:

```yaml
adaptive_kf: true
```

Use a custom tracker configuration when you have calibrated static Kalman
parameters against representative ground truth.

::: boxmot.BoostTrack
