# ReID Profiles

ReID configs live under `boxmot/configs/reid`.

## Role

Each file defines a reusable ReID profile with fields such as:

- `id`
- `model`
- optional `url`
- optional `device`
- optional `half`

## Example

```yaml
id: lmbn_n_duke
model: models/lmbn_n_duke.pt
url: https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth
device: ""
half: false
```

These profiles are referenced by benchmark configs and can also be selected explicitly from the CLI.
