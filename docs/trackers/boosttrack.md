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
