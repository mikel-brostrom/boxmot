# Add Catalog Entries and Experiments

Config additions should follow the existing split:

- `boxmot/configs/datasets/` for dataset facts
- `boxmot/configs/experiments/` for dataset + detector + ReID composition
- `boxmot/configs/detectors/` for detector profiles
- `boxmot/configs/reid/` for runtime ReID profiles
- `boxmot/trackers/<tracker>/config.py` for typed algorithm defaults and validation
- `boxmot/configs/trackers/<tracker>.yaml` for tuning metadata and component profiles
- `boxmot/configs/trackers/presets/` for tuned overrides

## Common change sets

Adding a new dataset/experiment combination usually means:

1. add a dataset YAML
2. add an experiment YAML
3. confirm detector and ReID profiles already exist or add them
4. use kebab-case IDs for ID-backed assets and kebab-case paths for experiments
5. update docs if the experiment becomes a documented workflow

Adding a tuned tracker usually means:

1. add or update a runtime preset without changing the tracker defaults
2. validate `track`, `eval`, and `tune`
3. document any new behavior or defaults

Tracker YAML algorithm entries declare tuning metadata such as `type`, `range`,
`options`, or conditional `activates`. The loader supplies each algorithm field's
`default` from its config class. List every algorithm field in built-in profiles:
give effective algorithm parameters valid search ranges, and mark fixed settings
with an empty mapping and a short explanation. Custom profiles may omit fields
to keep their typed defaults. Component groups retain profile defaults. Presets under
`presets/` are runtime overlays and should identify their target tracker.

Validate catalog and tracker-config changes with:

```bash
uv run --no-sync pytest tests/unit/configs tests/unit/trackers/test_tracker_registry.py tests/unit/engine/tuning/test_tracker_search_configs.py
```

## Python model-name autocomplete

After changing a detector/ReID profile, the pretrained ReID download catalog,
the tracker manifest, or the Ultralytics dependency,
regenerate the editor suggestions and commit the generated files:

```bash
uv run --no-sync python -m tools.generate_model_names
uv run --no-sync python -m tools.generate_model_names --check
```

The generator reads the packaged detector and ReID profiles, `TRAINED_URLS` in
`boxmot/reid/core/catalog.py`, the tracker manifest, and the installed Ultralytics
official asset inventory. ReID checkpoint suggestions use the exact filename
stems; training backbones without cataloged weights are excluded. Use the locked
environment with the `yolo` extra to regenerate it. The detector catalog records
the Ultralytics version and includes checkpoints supported by the box-producing
backend tasks; classification, semantic segmentation, and prompt-only SAM assets
are excluded. Profiles with multiple detector checkpoints produce explicit
`profile/checkpoint` suggestions. The resulting `_model_names.py` files contain
static literal types; imports do not read YAML or load models. The component
tests check freshness in CI, and `boxmot/py.typed` makes the annotations
available to editors using an installed package.

Tracker constructors expose their matching typed algorithm config through
`config=`. Keep field types, defaults, validation, and algorithm documentation in
that config class. The groups in `boxmot/trackers/common/constructor.py` describe
only runtime geometry/class selection and mask guidance. Update those groups when
runtime options change, keeping tracker-specific restrictions intact. Constructor
tests check the config and runtime contracts, public exports, and field discovery.

Kalman trackers expose `kalman: KalmanConfig | None` directly.
Keep timing and filter policies under `kalman`, covariance under `kalman.noise`,
and gain-suppression settings under `kalman.ams`, using dotted scalar paths
only inside engine search/configuration code. Use `flatten_tracker_options` and
`nest_tracker_options` for profile conversion; preserve resolved timing and any
class profiles when writing calibrated or tuned settings.

Keep constructor argument documentation directly on each public tracker's
`__init__` so editor tooltips show that tracker's options. Use one Google-style
`Args:` section covering every explicit parameter and `**kwargs`; keep the class
docstring for its overview and attributes. Describe supported inherited settings
under `**kwargs`, using names from that constructor's typed option group. Update
the docstring with signature changes and avoid advertising unsupported ReID,
geometry, timing, or Kalman settings. Check the tooltip contract with:

```bash
uv run --no-sync pytest tests/unit/trackers/test_constructor_typing.py
```
