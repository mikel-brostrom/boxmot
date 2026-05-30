# ReID Profiles

ReID settings are defined inline in benchmark YAMLs under `boxmot/configs/benchmarks`.

## Role

Each benchmark includes a ReID block with fields such as:

- `id`
- `model`
- optional `url`
- optional `device`
- optional `half`

## Example

```yaml
reid:
	id: lmbn_n_duke
	model: models/lmbn_n_duke.pt
	url: https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth
	device: ""
	half: true
	preprocess: resize
```

The benchmark ReID block provides defaults and can be overridden explicitly from the CLI.
