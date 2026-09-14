# ByteTrack

[Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

ByteTrack's main idea is simple: do not throw away low-confidence detections too early. The paper shows that a second association pass over lower-score boxes recovers occluded or partially visible objects and reduces fragmented tracks without adding much complexity. In practice, it is one of the strongest motion-only baselines because it stays fast while improving ID continuity.

## What BoxMOT Needs For ByteTrack

- Detector only. ReID features are not required.
- Supports both AABB and OBB detections in BoxMOT.
- Good default when you want a fast, strong baseline and already trust the detector.

## Optional McByte++ Mask Guidance

Python ByteTrack can use the temporal mask cue described in
[McByte++](https://github.com/tstanczyk95/McBytePlusPlus/tree/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9).
Official EdgeTAM propagates masks associated with track IDs and uses their
coverage to aid ambiguous or isolated box associations. The integration covers
this mask cue; the paper's long-term ReID and camera-motion modules are outside
its scope.

The same optional cue is available to the other eight Python box trackers.
See [temporal mask guidance](../tasks/masks.md#use-temporal-masks-in-association)
for the complete list, each tracker's association rules, and shared CLI usage.
The measurements below apply specifically to ByteTrack.

From a BoxMOT source checkout, install the official package:

```bash
uv sync --extra cpu --extra yolo --group mask-guidance
uv run --no-sync boxmot track --tracker bytetrack --tracker-backend python \
  --geometry aabb --source 0 --edgetam --mask-guidance-weights edgetam.pt \
  --mask-guidance-max-objects 32 --device cpu --show
```

Use `--extra cu130` instead of `--extra cpu` for CUDA. Repeat the selected
PyTorch extra and mask-guidance group on later syncs. The group pins
[facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM/tree/7711e012a30a2402c4eaab637bdb00a521302c91).
`--edgetam` enables guidance using the default `edgetam.pt` checkpoint. It
downloads the official full checkpoint into `./models/edgetam.pt`, verifies
its published SHA-256, and reuses it thereafter. Use
`--mask-guidance-weights` to select a different checkpoint.
An explicit existing checkpoint path is also accepted. The complete checkpoint
provides backbone weights, so a separate pretrained RepViT download is unnecessary.
Runtime works without the cloned EdgeTAM repository.

CPU and MPS run temporal inference in FP32. CUDA uses BF16 on supported devices
and FP32 otherwise. The optional upstream CUDA extension supplies mask hole
filling where available; effective postprocessing is recorded in evaluation
provenance. Inference runs without gradients. Guidance is off by default;
omitting `--edgetam` retains ordinary ByteTrack. Weights alone do not enable
guidance, and `--no-edgetam` keeps it disabled even when weights are supplied.

Webcams, RTSP and other supported URLs, videos, image directories, and
caller-supplied frame sources use the same causal propagation path:

```bash
uv run --no-sync boxmot track --tracker bytetrack \
  --source rtsp://camera.example/live --edgetam --mask-guidance-weights edgetam.pt \
  --device cuda:0 --show
```

The source is streamed. Each update propagates masks before association,
using the preceding frame's confirmed detection boxes as prompts. Supply the
image on every update, including empty detection batches. Keep frame dimensions
fixed within a sequence and call `reset()` before a new sequence or resolution.
Canonical `Frame` metadata must have increasing frame indices and a consistent
sequence ID; initial offsets and gaps are allowed. Throughput depends on the
detector, object count, EdgeTAM, and device.

ByteTrack guidance requires the Python backend, AABB geometry, IoU association, and
`per_class=False`. Canonical outputs remain `Tracks`; packed outputs remain
8-column AABB rows. Propagated masks are auxiliary association data.
`--segmentor` independently generates detection-aligned masks; see
[EdgeTAM mask generation](../tasks/masks.md#generate-masks-with-edgetam).
Matching standalone and temporal EdgeTAM weights, device, and precision share
one model within a live workflow, with separate inference states.

### Association behavior

Each ByteTrack pass keeps its normal acceptance threshold, including configured
`match_thresh` for high-confidence detections (`0.5` for the low-confidence pass
and `0.7` for unconfirmed tracks). Candidate classification uses the original
cost matrix:

- An admissible pair is ambiguous when another pair in its row or column is
  also admissible.
- A pair is isolated when neither endpoint has an admissible geometric partner.
  This is the conservative isolation interpretation used by BoxMOT.
- Clear one-to-one matches and their endpoints retain their original costs.

By default, for an eligible pair, the propagated mask must be nonempty, at least `0.90`
of its pixels must fall inside the detection box, and at least `0.05` of the
clipped box must be mask foreground. The box-fill fraction is then subtracted
from the existing cost, and the ordinary assignment solver runs at the same
threshold. Negative adjusted costs are valid. Tracks without usable masks
continue ordinary association. The mask coverage and fill thresholds, prompt
overlap gate, and identity cap are configurable in the tracker YAML and
searchable with `tune`; see [guidance settings](../tasks/masks.md#tracker-yaml-and-tuning).

### Comparison with the McByte++ implementation

The tracker integration was checked against both association variants in
[McBytePlusPlus at revision `be1bbc0`](https://github.com/tstanczyk95/McBytePlusPlus/tree/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9).
The reference applies masks after confidence fusion in the high-confidence and
confirmation passes, and directly to IoU costs in the low-confidence pass.
BoxMOT follows that order and uses the same LAPJV assignment solver. Default
coverage `0.90`, fill `0.05`, and stage thresholds `0.9`, `0.5`, and `0.7`
also agree. BoxMOT continues to honor configured `match_thresh`.

The published code and paper differ on isolation: both reference association
variants adjust only ambiguous pairs whose original cost already passes the
stage threshold. They contain no recovery branch for above-threshold pairs.
BoxMOT additionally implements the paper-inspired conservative isolation rule
described above. Consequently, guided BoxMOT results are not guaranteed to match
the reference tracker exactly.

| Integration detail | BoxMOT behavior |
| --- | --- |
| Clear matches | Protects their endpoints without the reference's debugging `+10` cost penalties; admissible assignments are unchanged. |
| Pixel overlap | Clips the actual box extent using floor/ceil bounds; the reference truncates `tlwh`, which differs for fractional or partly off-screen boxes. |
| Prompt timing | Uses the preceding frame's matched detection boxes after confirmation, as does the reference demo. New mask admissions reseed retained identities from their latest masks. |
| Lifetime | Advances on empty detection frames and immediately releases retired or evicted mask states. The reference demo skips updates when detector output is `None`, and mask removal is optional. |
| Tracker policy | Keeps BoxMOT's existing birth, duplicate-removal, and matching policies. The reference uses a birth threshold of `track_thresh + 0.1` and disables duplicate-track removal. |

The compact EdgeTAM runtime and identity cap are BoxMOT deployment choices.
The reference's optional mask refresh, conditional camera-motion changes, and
long-term re-identification are not enabled by this integration. Other Python
box trackers adapt the same mask cue to their existing association stages; see
[the tracker comparison](../tasks/masks.md#association-rules).

### Memory budget and Python configuration

`edgetam.max_objects` in the tracker YAML defaults to 32.
`--mask-guidance-max-objects` accepts a positive integer and overrides that value
when explicitly supplied.
It caps identities with propagation memory, including temporarily lost tracks.
Admission prioritizes already-guided visible identities, then promptable
visible identities, then the most recently observed lost identities. Track ID
breaks ties. Prompts blocked by the overlap gate consume no slot. Evicted and
retired identities release their mask state immediately; admission is
reconsidered each frame, and every identity continues box tracking.

The runtime processes objects individually on CPU, CUDA, and MPS while sharing
frame features. It retains at most two normalized frames, one frame-feature
cache, and shared positional constants. Each retained identity holds one
conditioning memory, six recent spatial memories, fifteen recent object
pointers, and one current CPU boolean mask. Historical logits and interactive
prompt bookkeeping are discarded. The cap bounds stored identity state;
full-resolution masks still scale with image size, and framework allocator
reservations can exceed live tensor storage.

Python callers can configure the budget and thresholds directly:

```python
from boxmot import ByteTrack
from boxmot.trackers import MaskGuidanceConfig

tracker = ByteTrack(mask_guidance=MaskGuidanceConfig(
    checkpoint="edgetam.pt",
    device="cpu",
    max_objects=32,
    min_coverage=0.90,
    min_fill=0.05,
    prompt_overlap=0.10,
))
```

`ByteTrack` and `create_tracker()` also accept a constructed `MaskGuidance`
component. Its `propagator=` constructor argument accepts a prebuilt
`EdgeTAMMaskPropagator`, whose `predictor=` argument can share an official model
with standalone segmentation. State belongs to that tracker or workflow;
there is no process-wide model cache.

### Tune guidance settings

The built-in ByteTrack YAML groups search ranges under `edgetam`, resolving to
`edgetam.min_coverage`, `edgetam.min_fill`,
`edgetam.prompt_overlap`, and `edgetam.max_objects`. Their defaults
remain `0.90`, `0.05`, `0.10`, and `32`. Enable guidance during tuning to search
them with ByteTrack's existing parameters:

```bash
boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker bytetrack --tracker-backend python \
  --edgetam --mask-guidance-weights models/edgetam.pt --device mps \
  --n-trials 20 --max-concurrent-trials 1 --sequence-workers 1
```

Guided trials keep IoU association. Without guidance, the mask parameters are
excluded from search. Pass a scalar `--tracker-config` profile to set the
starting values. Reuse the resulting `best.yaml` with `--tracker-config` in
`track` or `eval`, enable `--edgetam`, and select the same checkpoint with
`--mask-guidance-weights` if it differs from the default. Enablement and the
checkpoint are workflow inputs, separate from the saved tracker parameters. See
[guided tuning](../modes/tune.md#tune-mask-guidance) for build reuse and evaluation.

### Evaluate MOT17 Ablation With EdgeTAM

Evaluate, preview, and record one MOT17 ablation sequence with ByteTrack and
EdgeTAM on MPS:

```bash
uv run --no-sync boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --sequence MOT17-02-FRCNN --tracker bytetrack --tracker-backend python \
  --asso-func iou --edgetam --mask-guidance-weights models/edgetam.pt \
  --mask-guidance-max-objects 32 --device mps --sequence-workers 1 \
  --show --save
```

The checkpoint path assumes the setup above has downloaded `models/edgetam.pt`.
Use `edgetam.pt` to request that download on first use. Omit `--sequence` for
the full ablation split, or select `--device cpu` or `--device cuda:0` as needed.
`--show` and `--save` automatically display and record propagated masks tinted
by track ID. Masks appear after confirmation and next-frame propagation.
Identities without guidance have no propagated overlay; masks of retained lost
identities may remain visible during recovery. No extra mask display flag is
needed, and the overlays remain auxiliary to MOT box metrics.

The command materializes or reuses detections, then propagates EdgeTAM masks
during replay and reports box tracking metrics. To select existing cached
detections, add `--build BUILD_ID`; source image references must be available.
`--device` selects propagation hardware even with an existing build. Selected
frames, including ablation slices and `--fps` sampling, arrive in recorded
order. Guidance starts fresh for each sequence. Keep `--eval-masks` disabled
for this box association evaluation.

One sequence worker is the guidance default. Additional workers load their own
models and temporal states. `mask-guidance.json` records the official revision,
checkpoint hash, precision, postprocessing, identity cap, effective thresholds,
resolved tracker options, and named association behavior; these settings also
fingerprint the output directory.
To compare guidance with ordinary ByteTrack, replay the same `--build` and
tracker settings with `--edgetam` and `--no-edgetam`.
Omit `--show --save` when measuring tracking latency, since rendering and
recording add work to the replay.

### Validation tools

Run the real-checkpoint smoke, compact-versus-unpruned singleton parity, and
small synthetic-build evaluation checks from the development environment:

```bash
uv run --no-sync python -m tests.ci.mask_guidance_smoke \
  --checkpoint models/edgetam.pt --device cpu
uv run --no-sync python -m tests.ci.mask_guidance_parity \
  --checkpoint models/edgetam.pt --device cpu
uv run --no-sync python -m tests.ci.mask_guidance_eval_smoke \
  --checkpoint models/edgetam.pt --device cpu
```

To record process memory, live device allocations, allocator
reservations, and history counts from an existing build:

```bash
uv run --no-sync python -m tests.ci.mask_guidance_memory \
  --build runs/materializations/BUILD_ID --checkpoint models/edgetam.pt \
  --device mps --workers 1 --max-objects 32 --output runs/mask-memory
```

The diagnostic requires `psutil`, writes JSONL per sequence, and checks history
bounds on each frame. Use an empty output directory. `--max-frames N` limits
the run. `--trace-stages` synchronizes individual operations and affects timing;
`--flush-cache` measures releasing unused MPS buffers after each frame.
Ordinary evaluation leaves allocator caching to PyTorch. These diagnostics
measure the selected workload and device; full paper reproduction and a
universal edge-device frame rate are not claimed.

For a bounded comparison on identical cached detections, run:

```bash
uv run --no-sync python -m tests.ci.mask_guidance_compare \
  --build runs/materializations/BUILD_ID --checkpoint models/edgetam.pt \
  --sequence MOT17-02-FRCNN --split ablation \
  --ground-truth datasets/mot/MOT17/ablation/MOT17-02-FRCNN/gt/gt.txt \
  --device mps --frames 60 --max-objects 32 --output runs/mask-comparison
```

This diagnostic starts each variant in a fresh process and writes tracking
metrics, synchronized update latency, and memory samples to `comparison.json`
and per-variant files. It evaluates the selected pedestrian frames with
MOTChallenge distractor handling. Latency excludes decoding and detection;
memory peaks are sampled after updates and do not measure every temporary
allocation. Mask guidance adds inference cost, and accuracy gains must be
checked on the intended workload.

### Measured limitations

A 60-frame MOT17-02-FRCNN ablation replay on Apple MPS, using the same cached
detections, default ByteTrack settings, and a 32-identity cap, produced the
following results with Python 3.12.10 and PyTorch 2.12.1:

| Measurement | ByteTrack | With EdgeTAM |
| --- | ---: | ---: |
| HOTA | 61.36 | 60.21 |
| IDF1 | 74.84 | 74.73 |
| MOTA | 57.36 | 58.75 |
| Mean update latency | 3.06 ms | 814.58 ms |
| Sampled process RSS | 341.2 MiB | 851.0 MiB |
| Sampled live device memory | 0.0 MiB | 489.8 MiB |
| Sampled driver allocations, including caches | 0.5 MiB | 1382.7 MiB |

Latency excludes the first eight warmup updates, decoding, and detection.
Memory was sampled after updates. The guided run retained up to 182.4 MiB of
propagation state. A separate 40-frame spawned MPS replay with a four-identity
cap verified bounded history and zero live device allocations after sequence
cleanup; driver reservations can remain cached.

This short slice showed lower HOTA and IDF1 despite higher MOTA. It does not
establish an accuracy gain or real-time edge performance. Real-checkpoint
compact/unpruned parity passed on CPU and MPS, including reseeding and both
temporal retention boundaries. A 1,000-frame synthetic unit test covers identity
turnover and released storage. CUDA hardware and a full SportsMOT benchmark
were unavailable for this validation.

## Native C++ Backend

BoxMOT also ships a native C++17 ByteTrack implementation under `boxmot/native/cpp/trackers/bytetrack/`. It supports:

- cached `eval` and `tune` streamed through the live typed API
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detection layouts in the native tracker path

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker bytetrack --tracker-backend cpp
boxmot track --tracker bytetrack --tracker-backend cpp --source 0
```

::: boxmot.ByteTrack
