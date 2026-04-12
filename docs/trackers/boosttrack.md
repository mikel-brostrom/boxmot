# BoostTrack

[Paper: BoostTrack++: Using Tracklet Information to Detect More Objects in Multiple Object Tracking](https://arxiv.org/abs/2408.13003)

BoostTrack++ focuses on a neglected part of MOT pipelines: deciding which detections are worth trusting in the first place. The paper extends BoostTrack by using tracklet history to build a richer similarity score, then boosts low-confidence detections when past evidence suggests they are real objects. In practice, that improves recall and identity stability without giving up the online tracking-by-detection setup.

## What BoxMOT Needs For BoostTrack

- A detector and, by default, a ReID model for the full configuration.
- AABB detections only in BoxMOT.
- Best when low-confidence true positives are a recurring problem and you want stronger association scoring than plain IoU or Mahalanobis distance.

::: boxmot.trackers.boosttrack.boosttrack.BoostTrack
