# Tracker Overview

BoxMOT ships multiple tracker backends behind one interface.

## Implementation taxonomy

Internal packages are grouped by the representation maintained as tracker
state, not by every optional cue they consume:

- `boxmot/trackers/box`: AABB or OBB state. Appearance embeddings, camera
  motion, masks, or frames may still be optional association inputs.
- `boxmot/trackers/mask`: reserved for trackers whose primary state is an
  instance mask. The namespace currently has no implementation-specific base.
- `boxmot/trackers/multimodal`: multiple primary representations or model
  memory are fundamental to the method; Sam2Mot lives here.

Each registered implementation also declares immutable geometry and input
capabilities. The directory communicates ownership; capability metadata is the
source used by factories and pipelines for validation.

Representation family is independent of runtime backend. `box`, `mask`, and
`multimodal` classify tracker state and organize its Python domain code;
selecting `backend="cpp"` chooses a registered native implementation without
creating a fourth tracker family.

## Current tracker set

| Tracker | Uses ReID | Uses masks | OBB support | Native C++ live | Cached eval/tune |
| --- | --- | --- | --- | --- | --- |
| ByteTrack | No | No | Yes | Yes | Yes |
| BotSort | Yes | No | Yes | Yes | Yes |
| StrongSort | Yes | No | Yes | No | No |
| OcSort | No | No | Yes | Yes | Yes |
| DeepOcSort | Yes | No | Yes | No | No |
| HybridSort | Yes | No | Yes | No | No |
| BoostTrack | Yes | No | Yes | No | No |
| OccluBoost | Yes | No | Yes | Yes | Yes |
| SFSORT | No | No | Yes | Yes | Yes |
| [Sam2Mot](sam2mot.md) | No | Yes | Yes | No | No |

## How to choose

- Start with `bytetrack` when you want a fast motion-only baseline.
- Use `botsort`, `strongsort`, `deepocsort`, `hybridsort`, `boosttrack`, or `occluboost` when appearance cues matter.
- Use `sam2mot` when each detection has a row-aligned segmentation mask and you want mask-aware association without ReID.
- All registered Python trackers accept both AABB and OBB detections.
- All registered Python trackers expose the same selectable `asso_func`; see
  [tracker configuration](../config/trackers.md#association-function) for the
  supported AABB and OBB choices.
- Use `--tracker-backend cpp` for native C++ implementations when the selected tracker has a native backend.

## Config and factory

- Tracker runtime defaults and tuning search spaces share `boxmot/configs/trackers/<tracker>.yaml`; reusable scalar presets remain under `boxmot/configs/trackers/presets`.
- The runtime factory lives in `boxmot/trackers/factory.py`.
- Native C++ sources and low-level ctypes bindings live under `boxmot/native/`.
  Domain adapters live beside their algorithms at
  `boxmot/trackers/<family>/<name>/native.py`; box trackers therefore use
  `boxmot/trackers/box/<name>/native.py`.

Use [Native C++ Integration](../native/index.md) when you want to compile and embed a tracker directly in a C++ program.

Use the pages below for each tracker's API reference.
