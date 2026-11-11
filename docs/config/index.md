# Configuration

BoxMOT keeps packaged tracking configuration in the central `boxmot/configs`
catalog. Experiments describe reproducible dataset/model compositions.
Perception materialization requires a detector-based experiment; saved-detection
experiments instead read their dataset inputs directly. Tracker selection
remains a runtime choice.

## Config families

- `experiments/` selects a dataset split, detector profile, optional ReID profile, and class map.
- `datasets/` describes dataset facts and download locations.
- `detectors/` describes detector models and named checkpoints.
- `reid/` describes reusable runtime ReID models.
- `segmentors/` describes detection-aligned mask generators.
- `trackers/<tracker>.yaml` contains tracker runtime defaults and tuning search spaces.
- `trackers/presets/` contains reusable tracker overrides.
- `runtime.yaml` contains shared tracking-workflow defaults.

Use a detector-based experiment to build perception:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker boosttrack
```

The `--experiment` option accepts a catalog-relative YAML filename or an
explicit YAML path. Use the catalog-relative form for built-in experiments;
catalog assets use repository-relative paths rather than workstation-specific
absolute paths.

When perception choices should change, add a new experiment YAML that
references the desired dataset, split, detector checkpoint, and optional ReID
or segmentor config. Perception materialization has no dataset-only,
direct-source, or component-override mode. The separate
[`materialize --time-variant`](../modes/time-variant.md) operation derives a
frame-loss dataset from an existing build without running perception.

Before setup begins, BoxMOT expands experiment references, resolves numeric
class IDs, and validates the combination. Materialized manifests record the
resolved semantic component fingerprints; eval, tune, and research validate
them during replay. Eval and tune can first run perception when no build is
supplied; research always requires an existing build.

## Related pages

- [Mode Defaults](modes.md)
- [Experiments](experiments.md)
- [Supported datasets](../datasets/index.md)
- [Dataset profiles](datasets.md)
- [Detectors](detectors.md)
- [ReID Profiles](reid.md)
- [Masks and segmentors](../tasks/masks.md)
- [Tracker YAMLs](trackers.md)
