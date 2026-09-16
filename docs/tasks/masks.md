# Mask Tracking

Masks are canonical full-frame, detection-aligned `MaskBatch` values with
`bool[N,H,W]` CPU-contiguous storage.

```bash
boxmot track \
  --detector yolo11n-seg.pt \
  --geometry aabb \
  --tracker maf_hda \
  --per-class \
  --source video.mp4 \
  --save
```

[MafHda](../trackers/maf_hda.md) uses AABB detections, their masks, and the current
image to combine motion with masked correlation-filter appearance. Each detection
mask must contain foreground. Supply the image on every update, including frames
with no detections; OBB geometry is not supported by this tracker.

Masks can come from the detector or a standalone segmentor. SAM and official
EdgeTAM box prompting are available through `boxmot.segmentors`.

```python
from boxmot.structures import MaskBatch

enriched = detections.with_masks(
    MaskBatch(full_frame_masks_bool_cpu.contiguous())
)
tracks = tracker.update(enriched, frame=frame)
```

MafHda emits currently observed, confirmed tracks with aligned full-frame masks.
Lost tracks retain state for later recovery and are not emitted on missing
observations. Renderers prefer `Tracks.masks`; if absent, they map detection
masks through each nonnegative `detection_indices` value. A coasting row from a
box tracker with index `-1` has no current detection mask.

Standalone segmentors receive `(frames, detections)` and preserve detection
order. Requested empty outputs are present `bool[0,H,W]` tensors rather than
`None`, and empty inputs avoid model computation.

## Generate masks with EdgeTAM

The supported runtime uses the full PyTorch `.pt` checkpoint. The consolidated
EdgeTAM work also retains an unfinished TFLite export draft: `export-edgetam`
and TFLite bundle loading are not functional yet. The component exporter,
bundle manifest loader, and TFLite propagator implementations are still missing;
installing the `edgetam-export` extra alone does not enable them.

From a source checkout, install the optional official EdgeTAM package:

```bash
uv sync --extra cpu --extra yolo --group mask-guidance
uv run --no-sync boxmot track --source video.mp4 --detector yolo26n \
  --tracker bytetrack --segmentor boxmot/configs/segmentors/edgetam.yaml \
  --device cpu --save
```

Use `--extra cu130` instead of `--extra cpu` for CUDA. Repeat the selected
PyTorch extra and `--group mask-guidance` on later syncs. The supplied YAML
resolves the official full checkpoint into `models/edgetam.pt` and verifies its
SHA-256. Runtime does not require the cloned EdgeTAM repository.

The adapter uses the official `SAM2ImagePredictor`. It encodes one RGB frame,
prompts each detection box individually with `multimask_output=False`, and
releases image embeddings after that frame. OBB prompts use their enclosing
AABB. Outputs retain the detection order and full original image resolution.

The YAML defaults to `device: cpu`, `precision: fp32`, and
`options: {mask_threshold: 0.0}`. This is a native logit threshold, so any finite
value is valid; it is not a probability cutoff. CPU requires FP32; MPS supports
explicit FP32 or FP16. CUDA also supports FP16 and, on capable devices, BF16. An explicit live
`--device` overrides the YAML's device; precision remains authored in the YAML.

Python callers can resolve and construct the same component:

```python
from boxmot.segmentors import create_segmentor
from boxmot.segmentors.config import resolve_segmentor_spec

spec, provenance = resolve_segmentor_spec(
    "boxmot/configs/segmentors/edgetam.yaml", geometry="aabb"
)
segmentor = create_segmentor(spec)
masks = segmentor.segment([frame], [detections])[0]
enriched = detections.with_masks(masks)
```

Here `frame` is a canonical RGB `Frame`, and `detections` has its matching
`sample_id`. An injected official model can be passed with
`create_segmentor(spec, model=model)`; the artifact identity is still verified.

## Materialize detection masks

The `mot17/ablation-yolox-edgetam-lmbn.yaml` experiment combines YOLOX detections,
EdgeTAM box-prompted segmentation, and LMBN embeddings on the MOT17 ablation split:

```bash
uv run --no-sync boxmot materialize \
  --experiment mot17/ablation-yolox-edgetam-lmbn.yaml \
  --publish-masks --publish-embeddings \
  --device mps
```

Use the printed build ID with the same experiment's `eval --build BUILD_ID`.
The original `mot17/ablation-yolox-lmbn.yaml` experiment has no segmentor.
Select the segmentation experiment explicitly with `--experiment`; direct
dataset/detector/ReID selectors match experiments without a standalone segmentor.

Published masks are aligned to individual detections. Trackers that require
detection masks can consume them from the build. OccluBoost's `--edgetam`
temporal guidance computes its own track-ID masks during replay and does not
reuse these detection masks.

## Use temporal masks in association

The nine Python box trackers can propagate masks from preceding confirmed
detections: **ByteTrack, BotSort, StrongSort, OcSort, DeepOcSort, HybridSort,
BoostTrack, OccluBoost, and SFSORT**. Enable the same cue with
`--edgetam` and change `--tracker` to select the algorithm:

```bash
uv run --no-sync boxmot track --source video.mp4 \
  --tracker botsort --tracker-backend python --geometry aabb --asso-func iou \
  --edgetam --mask-guidance-weights edgetam.pt --mask-guidance-max-objects 96 \
  --device mps --save

uv run --no-sync boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker ocsort --tracker-backend python --asso-func iou \
  --edgetam --mask-guidance-weights edgetam.pt --mask-guidance-max-objects 96 \
  --device mps --sequence-workers 1
```

Guidance is off by default in `track`, `eval`, and `tune`. `--edgetam` uses the
default `edgetam.pt` checkpoint, resolved into `./models`; use
`--mask-guidance-weights PATH` to select another checkpoint. Checkpoint and cap
options alone do not enable guidance. `--no-edgetam` keeps it disabled even when a
checkpoint path is supplied.

Install the optional package using the command in
[mask generation](#generate-masks-with-edgetam). Use `--device cpu` or
`--device cuda:0` for those devices. Guidance requires **AABB geometry,
`asso_func: iou`, and `per_class: false`**. It is unavailable in native C++
backends, EagerMOT's 3D fusion, and MafHda's mask-based association. MafHda can
consume the standalone EdgeTAM detection masks described above.
Keep `--asso-func iou` when changing `--tracker`; HybridSort's authored default
uses DIoU, which this guidance integration does not support.

Propagation runs once before current-frame association, including frames with
no detections. Prompts come from the preceding frame's confirmed detection
boxes. Supply image pixels on every update and call `reset()` between sequences
or resolutions. Auxiliary masks do not require detection masks and preserve
canonical `Tracks` outputs and the eight-column packed AABB layout. Ordinary
tracking continues for identities without usable masks.

### View propagated masks

With guidance enabled, the existing `--show` and `--save` options automatically
overlay propagated masks, tinted by track ID. This works in both `track` and
`eval`; no additional display flag or standalone segmentor is needed.

Masks first appear after an identity is confirmed and propagation reaches the
next frame. Identities awaiting a prompt or outside the guidance budget have
no propagated overlay. Masks for retained lost identities may remain visible
while the tracker attempts recovery, until their mask state is retired or
evicted. The overlays show auxiliary association masks; MOT box metrics and
track output rows keep their existing meaning.

Without temporal guidance, detection and standalone segmentation masks retain
their existing rendering behavior. See the
[MOT17 preview and recording example](../trackers/bytetrack.md#evaluate-mot17-ablation-with-edgetam)
for a complete command.

### Association rules

Guidance classifies candidates before changing their costs. A pair is admissible
when its original cost is at or below the stage threshold, and ambiguous when
its row or column has multiple admissible partners. Only admissible, ambiguous
pairs receive mask adjustments; masks cannot rescue an above-threshold pair.
Clear one-to-one matches retain their own costs. As in McByte++, each clear match
adds `10` to its competing row and column costs; a competitor sharing both
protected endpoints receives both penalties. With guidance enabled, this
clear-match protection also applies when no usable masks are available.

By default, a mask must be nonempty, have at least `0.90` of its pixels inside
the clipped detection box, and fill at least `0.05` of that box. Its box-fill
fraction is subtracted from the association cost, or added to an equivalent
similarity, before the ordinary assignment solver runs. Pixel crops follow
McByte++: truncate the box's top-left coordinates, width, and height to integers,
clamp the origin to zero, then clip the crop to the image bounds. Empty or invalid
crops receive no mask adjustment.

The integration retains each algorithm's own stages and acceptance thresholds:

| Tracker | Where the cue is applied |
| --- | --- |
| ByteTrack | High-confidence, low-confidence, and confirmation costs; configured `match_thresh` stays effective. |
| BotSort | Final geometry/appearance costs, low-confidence costs, and confirmation costs, at their respective thresholds. |
| StrongSort | Appearance matching and geometric fallback; motion and stale-track hard gates remain enforced. |
| OcSort, DeepOcSort | Geometry similarity in initial association and observation recovery, plus OcSort's optional low-confidence pass; motion and appearance contributions remain. |
| HybridSort | Geometry support in initial, low-confidence, and observation recovery stages; confidence and appearance rules remain. |
| BoostTrack, OccluBoost | Geometry support in the existing combined score; OccluBoost recovery and low-confidence passes retain their appearance and track-age gates. |
| SFSORT | First-pass box-cue costs and low-confidence IoU costs, including dynamically adjusted thresholds. |

These are adaptations of the McByte++ mask cue to each tracker. ByteTrack's
confidence fusion, mask gating, and assignment order agree with the inspected
reference implementation. All supported trackers share the reference's
admissibility, ambiguity, clear-match protection, and pixel-crop rules while
keeping their own stages and scoring. This does not establish full tracker or
model-output parity. See [the reference comparison](../trackers/bytetrack.md#comparison-with-the-mcbyte-implementation)
for the inspected revision and remaining differences.

### Tracker YAML and tuning

Each supported tracker's built-in YAML groups the following runtime defaults
and search definitions under `edgetam`. The table shows their resolved option
paths:

| Tracker option | Default | Tuning search | Meaning |
| --- | --- | --- | --- |
| `edgetam.min_coverage` | `0.90` | Uniform `[0.50, 1.0]` | Minimum fraction of mask pixels inside the detection box. |
| `edgetam.min_fill` | `0.05` | Uniform `[0.01, 0.25]` | Minimum fraction of the clipped box covered by the mask. |
| `edgetam.prompt_overlap` | `0.10` | Uniform `[0.01, 0.50]` | Block a new prompt when a box with a lower bottom edge overlaps this fraction of its area or more. |
| `edgetam.max_objects` | `96` | Choice `[8, 16, 24, 32, 64, 96]` | Maximum identities retaining propagation state. |

OccluBoost uses a wider `min_fill` search range of `[0.01, 0.30]` and
`max_objects` choices of `[32, 64, 96, 128]`, with the same runtime defaults.

The built-in YAML entries include `default`, `type`, and `range` or `options`.
For a custom runtime profile passed with `--tracker-config`, use scalar values
inside the same group:

```yaml
asso_func: iou
edgetam:
  min_coverage: 0.90
  min_fill: 0.05
  prompt_overlap: 0.10
  max_objects: 16
```

These values configure guidance in `track`, `eval`, and `tune`; enabling it is
separate, through `--edgetam`. The group has no `enabled` field and does not
select a checkpoint. An explicit `--mask-guidance-max-objects N` overrides the
profile's cap; omitting it uses the resolved tracker value, which defaults to 96.

With `--edgetam`, tuning searches these four settings alongside the
ordinary tracker parameters and fixes association to IoU. A runtime profile
sets the starting values; searchable values can change in later trials. Without
guidance, the four settings are excluded from search. An explicit
`--mask-guidance-max-objects N` holds the cap fixed throughout tuning. Keep both
`--max-concurrent-trials 1` and `--sequence-workers 1` when tuning within an
edge-device memory budget. See [guided tuning](../modes/tune.md#tune-mask-guidance)
for commands and replaying `best.yaml`.

### Budget and model sharing

With guidance enabled, use `--mask-guidance-max-objects N` to cap identities holding propagation
state or set `edgetam.max_objects` in the tracker YAML; the default is 96.
The cap is shared across all association stages of
one tracker. It does not limit the total number of box tracks. Admission
prioritizes guided visible identities, then promptable visible identities,
then recently observed lost identities; evictions release mask state.

The runtime uses one model, shared frame features, and compact per-identity
state. Temporal inference on MPS loads FP16 weights, uses FP16 autocast, and
stores mask-memory features in FP16 automatically. CPU uses FP32; CUDA uses
BF16 autocast on supported devices and FP32 otherwise.

Propagation processes up to four objects with compatible temporal histories
per inference call and keeps their masks on the inference device. Matching
transfers only small integer overlap counts to the CPU; rendering creates CPU
masks lazily when requested. Reseeding reuses the device masks. Initial
prompts remain individual. This batch size is separate from the identity cap.
Each identity retains one conditioning memory, six recent spatial
memories, fifteen recent object pointers, and one current device boolean mask.
Full-resolution mask storage still depends on image size. Evaluation defaults
to one sequence worker; additional workers own separate models and state.

When a live workflow enables both standalone EdgeTAM and temporal guidance with
matching checkpoint, device, and precision, the adapters share one model with
separate inference state. Other model instances are scoped to their own run.

All supported Python constructors and `create_tracker()` accept
`mask_guidance=MaskGuidanceConfig(...)` or a constructed `MaskGuidance`:

```python
from boxmot.trackers import MaskGuidanceConfig, create_tracker

tracker = create_tracker(
    "botsort",
    asso_func="iou",
    mask_guidance=MaskGuidanceConfig("edgetam.pt", device="cpu"),
    edgetam={"max_objects": 16},
)
```

Python callers enable guidance by injecting the config or component. The
optional `edgetam` mapping overrides its parameter values; that mapping alone
does not enable guidance.

See [the ByteTrack guide](../trackers/bytetrack.md#optional-mcbyte-mask-guidance)
for checkpoint details, reset requirements, and diagnostic commands.
The published [short MOT17 measurements](../trackers/bytetrack.md#measured-limitations)
cover **ByteTrack only** and showed mixed accuracy with substantial MPS latency.
The other eight integrations have not established an accuracy or speed gain;
compare each guided tracker against its ordinary version on identical cached
detections before choosing it for deployment.
