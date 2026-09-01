# Add Catalog Entries and Experiments

Config additions should follow the existing split:

- `boxmot/configs/datasets/` for dataset facts
- `boxmot/configs/artifacts/` for public and precomputed data
- `boxmot/configs/experiments/` for dataset + detector + ReID composition
- `boxmot/configs/detectors/` for detector profiles
- `boxmot/configs/reid/` for runtime ReID profiles
- `boxmot/configs/trackers/<tracker>.yaml` for tracker defaults and tuning metadata
- `boxmot/configs/trackers/presets/` for tuned overrides

## Common change sets

Adding a new dataset/experiment combination usually means:

1. add a dataset YAML
2. add an experiment YAML
3. confirm detector and ReID profiles already exist or add them
4. use unique kebab-case IDs and portable repository-relative paths
5. update docs if the experiment becomes a documented workflow

Adding a tuned tracker usually means:

1. add or update a scalar preset without changing the tracker defaults
2. validate `track`, `eval`, and `tune`
3. document any new behavior or defaults

Tracker YAML files use the combined runtime/search schema. Each parameter entry
declares a scalar `default` plus tuning metadata such as `type`, `range`,
`options`, or conditional `activates`. Presets under `presets/` are scalar
overlays and should identify their target tracker.

Validate catalog and tracker-config changes with:

```bash
uv run --no-sync pytest tests/unit/configs tests/unit/trackers/test_tracker_registry.py tests/test_config.py
```
