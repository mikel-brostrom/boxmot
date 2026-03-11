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

## 10. Integrating OBB Support for New Trackers

When adding oriented bounding box (OBB) support, follow this generic implementation guide.

### Core requirements

- Set `supports_obb = True` on the tracker class.
- Keep `@BaseTracker.setup_decorator` enabled so detection shape can trigger OBB mode automatically.
- Reuse shared detection plumbing from:
  - `boxmot/trackers/basetracker.py`
  - `boxmot/trackers/detection_layout.py`
- Do not hardcode column indices if layout helpers already provide them:
  - `self.detection_layout.boxes(...)`
  - `self.detection_layout.confidences(...)`
  - `self.detection_layout.classes(...)`
  - `self.detection_layout.with_detection_indices(...)`

### Data contract

- Input detections:
  - AABB: `(x1, y1, x2, y2, conf, cls)` (6 columns)
  - OBB: `(cx, cy, w, h, angle, conf, cls)` (7 columns)
- Output tracks:
  - AABB: 8 columns
  - OBB: 9 columns `(cx, cy, w, h, angle, id, conf, cls, det_ind)`

### Implementation checklist

1) Split AABB and OBB parsing paths
  - Add explicit detection parsing/init branches for each mode.
  - Preserve `conf`, `cls`, and `det_ind` in both paths.

2) Use a motion model that supports OBB state
  - Keep AABB and OBB state/measurement handling explicit.
  - If OBB adds dimensions (for example angle), ensure `initiate`, `predict`, and `update` all use matching state sizes.

3) Keep mode-dependent predict/update logic
  - If velocity/state reset behavior differs between AABB and OBB, implement separate branches.
  - Avoid combining incompatible state assumptions in one path.

4) Wire OBB-aware association
  - Ensure association uses OBB geometry in OBB mode.
  - For IoU distance matching, pass `is_obb=self.is_obb` where applicable.
  - If using `self.asso_func`, verify the OBB association mode is selected in OBB mode.

5) Preserve geometry accessors for downstream consumers
  - Expose `xywha` in OBB mode.
  - Keep `xyxy` available as enclosing AABB for compatibility where needed.
  - Maintain `history_observations` and `id` for plotting and lifecycle logic.

6) Keep OBB plotting/history stable
  - Append post-update OBB geometry to `history_observations`.
  - If angles are used for plotting, add angle continuity handling to avoid wrap/flip artifacts.

7) Emit schema-correct outputs
  - AABB outputs must remain 8 columns.
  - OBB outputs must remain 9 columns in the exact order:
    `(cx, cy, w, h, angle, id, conf, cls, det_ind)`.

### Testing expectations

At minimum, add or update tests to cover:

- tracker accepts OBB detections
- tracker returns 9-column OBB outputs
- OBB association path uses oriented geometry
- OBB plotting/history path remains stable across frames

If shared OBB plumbing changes, also consider extending:

- `tests/unit/test_inference.py`
- `tests/unit/test_base_backend.py`

### Design rule

Use shared OBB plumbing for detection mode/layout and keep tracker-specific OBB internals limited to algorithm-specific motion and association details.
