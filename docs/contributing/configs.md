# Add Configs and Benchmarks

Config additions should follow the existing split:

- `datasets/` for dataset facts
- `benchmarks/` for dataset + detector + ReID selection
- `detectors/` for detector profiles
- `reid/` for ReID profiles
- `trackers/` for tracker defaults and tuning spaces

## Common change sets

Adding a new benchmark usually means:

1. add a dataset YAML
2. add a benchmark YAML
3. confirm detector and ReID profiles already exist or add them
4. update docs if the benchmark becomes a documented workflow

Adding a tuned tracker usually means:

1. update the tracker YAML
2. validate `track`, `eval`, and `tune`
3. document any new behavior or defaults
