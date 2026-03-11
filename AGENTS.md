# AGENTS.md – Working Guidelines for **BoxMOT**

> These instructions apply to all directories in this repository.  \
> Nested `AGENTS.md` files (if added later) override rules for their subtrees.

---

## 1. Environment & Tooling

### Python & `uv`

- Use **Python 3.11** (or the version configured in `pyproject.toml`).
- Install `uv` (safe to rerun even if present):

  ```bash
  pip install uv
  ```

- Install dependencies using the existing workflow:

  ```bash
  uv sync --all-extras --all-groups
  ```

- `uv` will create a `.venv` in the project root. Prefer running everything through `uv` so you don’t have to manage activation manually:

  ```bash
  # Generic command wrapper
  uv run <command> [args...]
  ```

#### Running with the package context

Always run Python entry points as modules from the repo root, not as loose scripts, so that `from boxmot...` imports work correctly:

```bash
# ✅ Good – uses package context
uv run python -m boxmot.engine.cli --help

# ❌ Avoid – can break imports (e.g., ModuleNotFoundError: boxmot)
python boxmot/engine/cli.py --help
PYTHONPATH=. python boxmot/engine/cli.py --help
```

If you really need to use the virtualenv directly:

```bash
source .venv/bin/activate
python -m boxmot.engine.cli --help
```

## 2. Workflow

- Create feature branches for work:

  ```bash
  git checkout -b codex/<short-topic>
  ```

- Keep changes focused: one logical change per PR / task.
- Follow the existing structure and conventions of the modules you touch.

## 3. Coding Conventions

- Prefer Python type hints and docstrings for any new or modified functions/classes.
- Keep imports:
  - Sorted.
  - Minimal (remove unused).
- Do not wrap imports in `try/except` unless there is a very specific reason and it’s clearly documented.

**Logging**
- Use the existing logger (e.g., `LOGGER`) rather than `print` in library code.
- It’s fine to `print` in CLI entry points when it improves UX, but prefer consistent logging style.

**Match the surrounding style**
- Naming, spacing, line wrapping, click option style, etc.
- Reuse helper patterns (e.g., decorators like `core_options`, shared parsing helpers).

## 4. CLI-Specific Guidelines

When editing `boxmot/engine/cli.py` or other CLIs:

- Group options logically (e.g., input, inference, output, display), but maintain backwards-compatible option names and defaults where possible.
- Prefer reusable decorators for option groups (`core_options`, `plural_model_options`, etc.).
- Use parsing helpers (e.g., `parse_tuple`, `parse_hw_tuple`) rather than ad-hoc parsing in every command.
- Keep help text accurate and concise; if you change behavior, update:
  - The option help strings.
  - Any CLI examples in `README.md`, `docs/`, or `examples/`.

When adding a new command:

- Reuse `make_args` to build argparse-like namespaces.
- Align with existing subcommands’ style (`track`, `generate`, `eval`, `tune`, `export`).

## 5. Commit & PR Expectations

Commit messages should start with one of:

- `feat:` – new feature
- `fix:` – bug fix
- `refactor:` – internal-only changes / cleanup
- `docs:` – documentation only
- `ci:` – CI / tooling changes
- `perf:` – performance improvements

Each commit should represent a coherent change; avoid mixing unrelated edits.

PR / task descriptions should include:

- A short summary of user-facing changes.
- A Testing section (see below).
- Any follow-up work or known limitations.

## 6. Testing & Verification

**What to run**

- Default: run the pytest suite from the repo root:

  ```bash
  uv run pytest
  ```

- If the full suite is too heavy, at least run the tests relevant to your change, e.g.:

  ```bash
  uv run pytest tests/test_cli.py
  uv run pytest tests/path/to/affected_module_tests.py
  ```

- When touching CLI / engine entry points, it’s useful to smoke-test common commands:

  ```bash
  uv run python -m boxmot.engine.cli --help

  # Example invocations (adjust source/paths as available in your env)
  uv run python -m boxmot.engine.cli track --source <path-or-url> ...
  uv run python -m boxmot.engine.cli generate --source <path-or-url> ...
  uv run python -m boxmot.engine.cli eval --source <path-or-url> ...
  uv run python -m boxmot.engine.cli tune --source <path-or-url> ...
  ```

**If tests or commands cannot be run**

Sometimes the provided environment is missing GPUs, large datasets, or external services. In that case:

1. Try the following first:

   ```bash
   uv sync --all-extras --all-groups

   uv run python -m boxmot.engine.cli --help

   uv run pytest
   ```

2. If something still fails for reasons outside your control (e.g., missing CUDA runtime, no network for model downloads, etc.), do not fake test results. Instead, document clearly in your Testing section, for example:

   ```text
   Testing
   - uv run python -m boxmot.engine.cli --help  ✅
   - uv run pytest ❌ (not run)

   Reason: pytest requires GPU / CUDA dependencies that are not available in the current container.
   Please run `uv sync --all-extras --all-groups` and `uv run pytest` in a fully configured environment.
   ```

- Include the exact commands you ran and a brief reason why anything couldn’t be completed.

## 7. Documentation & Examples

- Update docs or examples when behavior or interfaces change, especially:
  - CLI options or defaults.
  - New or removed commands.
- Keep README snippets and CLI help text in sync with code updates.
- When changing data formats or output directories, update any references in:
  - `docs/`
  - `examples/`
  - `tests/`

## 8. Performance & Safety

- Be mindful of model weights and large assets:
  - Do not commit generated artifacts or large binaries.
  - Prefer referencing weights via URLs or documented download steps.
- Where practical:
  - Use deterministic or seeded behavior for tests/examples.
  - Avoid unnecessary heavy computation in unit tests.

## 9. Integrating a New Tracker (Checklist)

1) Implement the tracker
  - Add a new module under `boxmot/trackers/<name>/` (e.g., `sfsort.py`).
  - Implement a tracker class that subclasses `BaseTracker` and defines `update()`.

2) Register the tracker
  - Add the tracker to `TRACKER_MAPPING` in `boxmot/trackers/tracker_zoo.py`.
  - Export it in `boxmot/trackers/__init__.py` and `boxmot/__init__.py`.
  - Add the tracker name to the `TRACKERS` list in `boxmot/__init__.py`.

3) Add default configuration
  - Create `boxmot/configs/trackers/<name>.yaml` with default parameters and tuning ranges.

4) Update docs
  - Add a tracker doc page in `docs/trackers/<name>.md`.
  - Add the tracker to `mkdocs.yml` nav.
  - Mention it in `docs/index.md` and `README.md` where trackers are listed.

5) Update tests
  - Register the tracker in `tests/test_config.py` lists so it’s covered by unit tests.

6) Update CI/benchmarks
  - Add the tracker name to workflow matrices/lists in `.github/workflows/`.

7) Commit new files
  - Ensure new tracker code, config, and docs are staged and pushed.

## 10. OBB Integration Playbook (BotSort, ByteTrack, OCSort)

Use this as the canonical implementation pattern when adding OBB support to another tracker.

### Shared OBB contract (all trackers)

- Declare `supports_obb = True` on the tracker class.
- Reuse shared mode/layout plumbing in:
  - `boxmot/trackers/basetracker.py`
  - `boxmot/trackers/detection_layout.py`
- Do not hardcode OBB column indices when layout helpers already provide them:
  - `self.detection_layout.boxes(...)`
  - `self.detection_layout.confidences(...)`
  - `self.detection_layout.classes(...)`
  - `self.detection_layout.with_detection_indices(...)`
- Expect input shapes:
  - AABB detections: `(x1, y1, x2, y2, conf, cls)` (6 cols)
  - OBB detections: `(cx, cy, w, h, angle, conf, cls)` (7 cols)
- Output shapes must stay consistent with `DetectionLayout`:
  - AABB output: 8 cols
  - OBB output: 9 cols `(cx, cy, w, h, angle, id, conf, cls, det_ind)`
- Association mode naming is automatic via layout:
  - base `"iou"` becomes `"iou_obb"` in OBB mode.
- Keep plotting compatibility:
  - Track objects should expose `xywha`, `xyxy`, `history_observations`, `id`.

### BotSort OBB reference (`boxmot/trackers/botsort/`)

- Tracker-level behavior:
  - Uses `KalmanFilterXYWH(ndim=self.detection_layout.box_cols)` in `update()`.
  - Disables camera motion compensation in OBB mode (`self.cmc = None`).
  - Uses `self.detection_layout` for detection splitting and confidence indexing.
  - Passes `is_obb=self.is_obb` in all IoU matching calls.
- Track object behavior (`botsort_track.py`):
  - Keeps explicit init paths:
    - AABB: `xyxy -> xywh`
    - OBB: `det[:5] -> xywh` (where `xywh` holds `xywha` in OBB mode)
  - Uses separate shared Kalman instances:
    - AABB: `KalmanFilterXYWH()`
    - OBB: `KalmanFilterXYWH(ndim=5)`
  - Prediction zeroing differs by mode:
    - AABB resets `[6:8]`
    - OBB resets `[7:10]` (size/angle velocities)
  - Maintains OBB-specific plotting history via `_state_obb_for_plot()`:
    - swaps `w/h` when needed
    - unwraps angle with periodic wrapping
    - stores flattened 4-corner polygons (8 values)
  - `xyxy` for OBB is enclosing AABB from `cv2.boxPoints`.
  - Final output uses `t.xywha` in OBB mode.

### ByteTrack OBB reference (`boxmot/trackers/bytetrack/`)

- Tracker-level behavior:
  - Switches motion model in `update()`:
    - AABB: `KalmanFilterXYAH()`
    - OBB: `KalmanFilterXYWH(ndim=5)`
  - Uses layout helpers for detection indexing and confidence filtering.
  - Uses OBB-aware IoU in all association stages with `is_obb=self.is_obb`.
  - Builds outputs with `t.xywha` in OBB mode.
- Track object behavior (`bytetrack.py::STrack`):
  - AABB init stores `xywh`, `tlwh`, `xyah`.
  - OBB init stores 5D geometry in `xywh`, sets `tlwh/xyah=None`.
  - Activation/update/re-activation choose measurement by mode:
    - OBB: `xywh` (5D)
    - AABB: `xyah` (4D)
  - Uses separate shared Kalman filters:
    - AABB `shared_kalman = KalmanFilterXYAH()`
    - OBB `shared_kalman_obb = KalmanFilterXYWH(ndim=5)`
  - Keeps the same OBB plotting/corner-history strategy as BotSort.

### OCSort OBB reference (`boxmot/trackers/ocsort/ocsort.py`)

- Core difference from BotSort/ByteTrack:
  - OCSort uses an XYSR-state Kalman model, not XYWH/XYAH.
- OBB state mapping:
  - `convert_obb_to_z([cx, cy, w, h, theta]) -> [cx, cy, s, r, theta]`
  - `convert_x_to_obb([x, y, s, r, theta, ...]) -> [x, y, w, h, theta]`
- Kalman dimensions:
  - AABB: `dim_x=7, dim_z=4`
  - OBB: `dim_x=9, dim_z=5` (adds angle and angle velocity)
- OBB-specific tracking logic:
  - `k_previous_obs(..., is_obb=True)` returns 6-element placeholders.
  - Uses `speed_direction_obb` (center-to-center velocity direction).
  - On update, OBB path calls `kf.update(convert_obb_to_z(bbox[:5]))`.
  - OBB plotting history uses state-derived corners via `_state_obb_for_plot()`.
  - Prediction/state retrieval for OBB uses `convert_x_to_obb(...)`.
- Association/output:
  - Uses layout-driven slices (`box_cols`, `box_with_conf_cols`, `cls_idx`).
  - Uses `self.asso_func` from `BaseTracker`; in OBB mode this becomes `*_obb`.
  - Output includes 5 box values + `[id(+1 in current OCSort code), conf, cls, det_ind]` => 9 columns.

### Required checklist for adding OBB to another tracker

1) Declare capability and use shared mode inference
  - Set `supports_obb = True`.
  - Keep `@BaseTracker.setup_decorator` active so detection shape can trigger OBB mode.

2) Add explicit AABB vs OBB detection parsing
  - Implement separate parse/init branches in the track object.
  - Always preserve `conf`, `cls`, `det_ind`.

3) Use an OBB-capable motion state
  - Either follow BotSort/ByteTrack (`KalmanFilterXYWH(ndim=5)`) or OCSort-style mapped state (`[x,y,s,r,theta]`), depending on algorithm.

4) Keep mode-dependent predict/update measurement paths
  - Avoid one-size-fits-all updates if AABB and OBB states differ.

5) Wire OBB-aware association
  - For IoU matching paths, use `iou_distance(..., is_obb=self.is_obb)` or ensure `self.asso_func` resolves to OBB variants.

6) Preserve plotting/history semantics
  - For OBB, append 4-corner history (8 values) from the post-update state.
  - Keep angle continuity logic to avoid 90-degree flip artifacts.

7) Emit correct output schema
  - AABB: 8 cols, OBB: 9 cols.
  - For OBB output use `(cx, cy, w, h, angle, id, conf, cls, det_ind)`.

8) Add/extend tests
  - In `tests/unit/test_trackers.py` at minimum:
    - tracker accepts OBB detections
    - tracker returns 9-column OBB output
    - OBB matching uses oriented geometry
    - OBB history/plotting path remains stable
  - If shared plumbing changed, also extend:
    - `tests/unit/test_inference.py`
    - `tests/unit/test_base_backend.py`

### Design rule

Mirror shared OBB plumbing from `BaseTracker`/`DetectionLayout`, and copy tracker-specific internals from the closest existing pattern:

- BotSort/ByteTrack pattern: XYWH + 5D Kalman in OBB mode.
- OCSort pattern: XYSR(+theta) mapped OBB state.

Do not mix patterns unless the algorithm requires it.
