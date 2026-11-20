# Testing

From the repo root, the default test command is:

```bash
uv run --no-sync pytest
```

When a full run is too heavy, at least run the tests relevant to your change:

```bash
uv run --no-sync pytest tests/unit/engine/research/test_engine_research.py
uv run --no-sync pytest tests/unit/engine/test_cli_v24.py
uv run --no-sync pytest tests/unit/pipelines tests/unit/structures
uv run --no-sync pytest tests/unit/trackers
uv run --no-sync pytest tests/unit/configs
```

Native wrapper changes are covered under `tests/unit/native`; those tests may
need CMake, OpenCV, Eigen, and a working C++ compiler. Run native tests serially:
their fixtures build targets in shared directories. To parallelize the rest of
the suite as the release workflow does:

```bash
uv run --no-sync pytest -n logical --maxprocesses=4 --dist worksteal --ignore=tests/unit/native tests
uv run --no-sync pytest tests/unit/native
```

## CI smoke tests

### Helm and Docker images

Check chart rendering and CPU/GPU, multi-camera, external-service and invalid
configuration contracts without connecting to a Kubernetes cluster:

```bash
uv run --no-sync python -m tests.ci.helm_contract
```

The Helm workflow runs these checks on relevant PRs. The
[end-to-end harness](https://github.com/mikel-brostrom/boxmot/blob/master/tests/e2e/README.md)
exercises real tracking and attribute inference, installation, upgrades,
recovery, exports and retained storage in an isolated namespace. GPU execution
requires a suitably configured host. Docker image CI separately builds and
smoke-tests the CPU CLI and HTTP service targets before publication.

### Attribute database contracts

Attribute session tests use a real PostgreSQL database. Install the `attributes`
extra in addition to your selected CPU/CUDA profile and test dependencies, then
start the database and set the test connection URL:

```bash
docker run --detach --name boxmot-test-postgres \
  --publish 127.0.0.1:55432:5432 \
  --env POSTGRES_USER=boxmot --env POSTGRES_PASSWORD=boxmot-test \
  --env POSTGRES_DB=boxmot_test postgres:17

docker exec boxmot-test-postgres pg_isready -U boxmot -d boxmot_test
export BOXMOT_TEST_POSTGRES_URL='postgresql://boxmot:boxmot-test@127.0.0.1:55432/boxmot_test'
export BOXMOT_REQUIRE_POSTGRES_TESTS=1
uv run --no-sync python -m pytest tests/unit/attributes tests/unit/engine/test_attributes.py -q
```

Wait for `pg_isready` to report that the database accepts connections; repeat
it while initialization is in progress. Use a development database. Fixtures
remove only runs created by each test, so parallel pytest workers can share it.
When finished, remove this disposable test container and its anonymous data
volume with `docker rm --force --volumes boxmot-test-postgres`.

PostgreSQL-dependent tests normally skip when `BOXMOT_TEST_POSTGRES_URL` is unset.
`BOXMOT_REQUIRE_POSTGRES_TESTS=1` makes a missing URL or missing attribute client
dependencies fail at startup. PR and release validation set it with their
PostgreSQL service so required database coverage cannot silently disappear.
Config, crop, and HTTP transport tests do not require a database.

### Python API

Run the same isolated Python API smoke test used by the `python_api` CI job:

```bash
uv run --no-sync python -m pytest -p no:cacheprovider -q -s tests/ci/python_api_smoke.py
```

## CLI smoke tests

For CLI or engine entry-point changes, verify help and at least one representative command:

```bash
uv run --no-sync python -m boxmot.engine.cli --help
uv run --no-sync python -m boxmot.engine.cli track --help
uv run --no-sync python -m boxmot.engine.cli eval --help
```

For documentation changes, reproduce the docs workflow's warning-as-error
build:

```bash
.venv/bin/ruff check docs/gen_ref_pages.py tests/unit/docs
.venv/bin/ruff format --check docs/gen_ref_pages.py tests/unit/docs
.venv/bin/pytest -q tests/unit/docs
.venv/bin/mkdocs build --strict
```

The build runs `docs/gen_ref_pages.py` automatically. The script parses the
`boxmot` source tree without importing it, creates virtual API pages, and feeds
them to mkdocstrings. Generated reference Markdown is therefore not committed;
adding or removing a Python module is reflected by the next docs build.
See [Build the Documentation](documentation.md) for fresh environment setup,
live preview, and code-reference details.

## Document constraints

If GPU runtimes, datasets, or network downloads are unavailable, document exactly what you ran and what prevented fuller validation.
