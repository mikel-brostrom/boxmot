# CI and Benchmarks

When a change affects benchmarked trackers or supported tracker lists, check the workflow matrices under `.github/workflows`.

## Workflow triggers

The top-level `on:` block decides whether any job in a workflow is created:

| Workflow | Current triggers |
| --- | --- |
| `.github/workflows/ci.yml` | Pushes to `master` and pull requests targeting `master` |
| `.github/workflows/docs.yml` | Pushes to every branch and manual dispatch; deployment only on `master` |
| `.github/workflows/benchmark.yml` | Pushes to `main`, pull requests targeting `main`, and manual dispatch |

!!! warning "Benchmark branch mismatch"
    This repository's integration branch is currently `master`, while
    `benchmark.yml` filters automatic push and pull-request runs to `main`.
    Unless the workflow trigger is aligned, that workflow only runs through
    manual dispatch in this repository.

A valid job block does not run when its workflow was not triggered. In
particular, a direct push to a feature branch does not start `ci.yml`; open or
update a pull request targeting `master`, or push the commit to `master` through
the normal merge flow. There are currently no path filters in `ci.yml`.

The `python_api` job has no job-level `if:` or `needs:` condition. Once
`ci.yml` is triggered, it installs `.[yolo]` plus the test group and runs:

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q tests/ci/python_api_smoke.py
```

If that job is absent from a run that otherwise matches the trigger, confirm
that the workflow revision containing the job is part of the tested commit.

## What the main workflow covers

`ci.yml` separates tracker smoke tests, native builds and live backends, tuning,
metric parity/evaluation, ReID training, OBB, pose/detection/segmentation
integrations, export runtimes, the Python API smoke test, and the full pytest
suite. Expensive integration jobs run on Python 3.12; a smaller compatibility
matrix checks the supported 3.10 and 3.13 boundaries, while detector coverage
also exercises 3.11. The final `check-failures` job collects their results.

Python and uv setup is centralized in `.github/actions/setup-ci-python`, which
pins both actions and the uv release and enables uv's dependency cache.
The uv version has a single source of truth in `.github/uv-version`; give each
distinct extras/group combination a matching `cache-profile` so one partial uv
cache cannot shadow another job's dependency set.
Dependency installation is centralized in `.github/scripts/uv_ci_install.sh`;
it verifies that `uv.lock` is current and uses the lock as constraints before
creating the CPU-only environment. This preserves CPU-specific PyTorch wheels
without letting the remaining dependency versions drift. Pass the smallest
project extras and uv groups that the job imports. For example, the Python API
smoke job needs `yolo` plus `--group test`, while the docs job installs
`--group docs` and runs `uv run mkdocs build --strict`.

Detector and ReID checkpoints used directly by CI are prepared through
`.github/actions/prepare-ci-assets`. The action restores its model cache, then
verifies every file against a pinned SHA-256 digest before exposing its absolute
path through `BOXMOT_CI_*` environment variables. Add new network-loaded model
assets there instead of relying on a runtime download inside a test.

## Typical CI-sensitive changes

- adding a new tracker
- renaming tracker identifiers
- changing experiment IDs used by benchmark jobs
- modifying default tracker sets used in benchmark tables or matrices
- changing ReID, mask, OBB, or native-backend requirements

## Keep docs and CI aligned

If a tracker is exposed in the docs as supported, make sure the relevant tests
and workflow coverage reflect that support level. In particular, inspect the
shared `BOXMOT_CI_TRACKERS` and `BOXMOT_CI_REID_TRACKERS` lists and the
`BOXMOT_CI_CPP_TRACKERS` job list in `ci.yml`, plus the explicit tracker/backend
matrix in `benchmark.yml`. Every tracker in `BOXMOT_CI_TRACKERS` is required to
pass the OBB smoke test. Mask-aware trackers may need a dedicated mask source
or model instead of the generic bounding-box smoke command.

The native tracker smoke exercises ReID on Linux. macOS still covers native
build/load/tracking behavior, but skips ReID because exporting the required
ONNX sibling is intentionally disabled on GitHub's macOS runners.
