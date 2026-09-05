# ReID Profiles

Runtime ReID profiles live under `boxmot/configs/reid`.

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

`device: auto` inherits the command's selected device. `precision` is explicit
and accepts `fp16`, `fp32`, or `bf16`; the resolver adapts these settings to the
existing runtime arguments. `preprocessing.mode` is required, and
`preprocessing.image_size` must contain exactly two positive integers in
height-width order.

During materialization the engine resolves the artifact and SHA-256 before the
build ID is established. Python callers use the resulting values in a frozen
`ReIDEncoderSpec`; trackers receive only the produced embeddings and never the
profile or model instance.

## Related pages

- [ReID Models](reid-models.md)
