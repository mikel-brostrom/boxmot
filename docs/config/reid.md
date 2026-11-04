# ReID Profiles

Runtime ReID profiles live under `boxmot/configs/reid`. They select trained
weights and crop preprocessing; [training recipes](../modes/train.md) are
separate configurations under `boxmot/reid/training/configs/recipes`.

```yaml
id: lmbn-n-duke

weights:
  path: models/lmbn_n_duke.pt
  uri: https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_duke.pt

runtime:
  device: auto
  precision: fp16

preprocessing:
  mode: resize
  image_size: [384, 128]
```

CLI workflows select the runtime device; standalone profile resolution maps
`device: auto` to CPU unless you supply an override. Use `fp16` or `fp32`
for built-in encoders. Although the profile schema accepts `bf16`, the built-in
ReID adapters reject BF16 inference. Live `track` uses `--device` and `--half`
for device and precision, overriding the resolved profile precision.

`preprocessing.mode` is required and supports `resize` or `resize_pad`.
`resize_pad` preserves aspect ratio and pads with the ImageNet mean;
`resize` stretches to the configured crop. Match the training policy.
`preprocessing.image_size` must contain exactly two positive integers in
height-width order.

Crop extraction follows detection geometry automatically: AABB detections use
clipped axis-aligned crops and OBB detections use the canonical rectified OBB
transform. Built-in ReID profiles do not consume detection masks; a custom
mask-dependent encoder declares that requirement through its encoder contract.

During materialization the engine resolves the artifact and SHA-256 before the
build ID is established. Python callers use the resulting values in a frozen
`ReIDEncoderSpec`. The live tracking workflow gives that complete spec to a
ReID-enabled tracker adapter, which constructs its encoder only when a non-empty
update arrives without embeddings. For a direct real-time loop, pass a grouped
`ReIDConfig` through the tracker's `reid` argument:

```python
from boxmot import BotSort, ReIDConfig
from boxmot.reid import create_reid_encoder

config = ReIDConfig(model="osnet-x0-25-msmt17", device="cpu", precision="fp32")
tracker = BotSort(reid=config)
encoder = create_reid_encoder(config)
```

`ReIDConfig` groups the model selection, runtime device and precision,
preprocessing, image size, embedding dimension, and batch size. Optional
overrides preserve profile defaults when omitted. `use_embeddings` and
association, smoothing, and gallery settings remain tracker parameters.
Pass a prebuilt `AppearanceEncoder` as `reid=encoder` to reuse it, or install a
resolved `ReIDEncoderSpec` with `tracker.configure_reid(spec)`. A separately
composed `AppearanceEncoder` remains available when embeddings should be shared
or exposed as pipeline output. Native adapters support the same lazy path;
their low-level C++ tracker libraries remain model-free consumers of embedding
buffers.

The encoder processes at most 64 crops per batch by default; set
`ReIDConfig(batch_size=...)` to adjust it. `embedding_dim` is an optional
declared output width, not a request to resize a descriptor. Provide it for
custom runtimes whose output dimension cannot be inferred before an empty
batch is encoded. `allow_download=False` requires existing local weights.

Model references can also select exported `.onnx`, `.engine`, `.torchscript`,
or `.tflite` files, an `_openvino_model` directory, or a `_coreml_model` bundle.
The suffix selects the runtime. Install the corresponding optional dependency
and preserve export metadata beside the artifact; see [Export](../modes/export.md).

## Related pages

- [ReID Models](reid-models.md)
