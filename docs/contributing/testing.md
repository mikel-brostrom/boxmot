# Testing

From the repo root, the default test command is:

```bash
uv run pytest
```

When a full run is too heavy, at least run the tests relevant to your change:

```bash
uv run pytest tests/unit/test_engine_research.py
uv run pytest tests/test_cli.py
```

## CLI smoke tests

For CLI or engine entry-point changes, verify help and at least one representative command:

```bash
uv run python -m boxmot.engine.cli --help
uv run python -m boxmot.engine.cli track --help
uv run python -m boxmot.engine.cli eval --help
```

## Document constraints

If GPU runtimes, datasets, or network downloads are unavailable, document exactly what you ran and what prevented fuller validation.
