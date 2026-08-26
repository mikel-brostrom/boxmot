# Testing

From the repo root, the default test command is:

```bash
uv run pytest
```

When a full run is too heavy, at least run the tests relevant to your change:

```bash
uv run pytest tests/unit/engine/research/test_engine_research.py
uv run pytest tests/unit/engine/test_cli.py
uv run pytest tests/unit/api/test_python_api.py
uv run pytest tests/unit/trackers
uv run pytest tests/unit/configs
```

Native wrapper changes are covered under `tests/unit/native`; those tests may
need CMake, OpenCV, Eigen, and a working C++ compiler.

## CI smoke tests

Run the same isolated Python API smoke test used by the `python_api` CI job:

```bash
uv run python -m pytest -p no:cacheprovider -q -s tests/ci/python_api_smoke.py
```

## CLI smoke tests

For CLI or engine entry-point changes, verify help and at least one representative command:

```bash
uv run python -m boxmot.engine.cli --help
uv run python -m boxmot.engine.cli track --help
uv run python -m boxmot.engine.cli eval --help
```

For documentation changes, reproduce the docs workflow's warning-as-error
build:

```bash
uv run mkdocs build --strict
```

## Document constraints

If GPU runtimes, datasets, or network downloads are unavailable, document exactly what you ran and what prevented fuller validation.
