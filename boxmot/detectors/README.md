# Detector components

`boxmot.detectors` is a domain package. A detector receives canonical RGB
frames and returns one aligned canonical detection collection per frame:

```python
Detector.predict(frames: Sequence[Frame]) -> list[Detections]
```

It does not open sources, iterate streams, render output, run callbacks, or own
CLI workflow state.

## Public construction

```python
from boxmot.detectors import DetectorSpec, create_detector

detector = create_detector(
    DetectorSpec(
        backend="ultralytics",  # ultralytics, yolox, or rtdetr
        artifact=resolved_path,
        artifact_sha256=resolved_sha256,
        device="cuda:0",
        precision="fp16",
        preprocessing="default",
        geometry_mode="aabb",
        options=(("confidence", 0.25),),
    )
)
detections = detector.predict(frames)
```

Resolve and hash artifacts before construction when the component participates
in materialization. `DetectorSpec.options` is a sorted tuple containing only
immutable JSON values.

## Contract

Every backend exposes a frozen `DetectorCapabilities` value declaring mask and
embedding output plus AABB/OBB support. Declarations provide early validation;
pipelines still validate each actual result.

`Frame.image` is CPU-contiguous RGB `torch.uint8[3,H,W]`. Backends may move
private inference tensors to an accelerator, but every returned structure is
CPU-contiguous:

- AABB geometry: `Boxes(float32[N,4])` in `xyxy` order.
- OBB geometry: `OrientedBoxes(float32[N,5])` in `cxcywha` order with radians.
- Scores: `float32[N]` in `[0,1]`.
- Classes: nonnegative `int64[N]`.
- Optional masks: full-frame `bool[N,H,W]`.
- Optional embeddings: `float32[N,D]`.

Backends must preserve input order and `sample_id`. An empty result retains its
geometry mode and returns correctly shaped empty tensors without model work
when the whole input batch is empty.

## Adding a backend

1. Implement the runtime `Detector` protocol in
   `boxmot/detectors/backends/<name>.py`.
2. Convert framework values explicitly at its boundary; canonical constructors
   never cast, move, clip, or filter values.
3. Give the backend class (or a dedicated factory) a constructor accepting
   exactly one `DetectorSpec`.
4. Register its lazy `module:callable` reference in the shared registry declared
   by `boxmot/detectors/factory.py`.
5. Add tests for AABB/OBB support, empty batches, ordering, IDs, masks, and lazy
   imports.

Keep training code with its model domain and keep source, sink, service,
materialization, retry, and display concerns in `boxmot.engine`.
