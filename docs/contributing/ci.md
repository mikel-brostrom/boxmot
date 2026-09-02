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
`ci.yml` is triggered, it installs the CPU profile, `yolo` extra, and test
group, then runs:

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q -s tests/ci/python_api_smoke.py
```

If that job is absent from a run that otherwise matches the trigger, confirm
that the workflow revision containing the job is part of the tested commit.

## What the main workflow covers

`ci.yml` separates tracker smoke tests, native builds and live backends, tuning,
metric parity/evaluation, ReID training, OBB, pose/detection/segmentation
integrations, export runtimes, the Python API smoke test, and the full pytest
suite. The final `check-failures` job collects their results.

The local `.github/actions/setup-ci-python` action installs Python and the uv
version required by the root `pyproject.toml`, and enables uv's dependency
cache. Each job then runs a locked sync and explicitly selects exactly one
PyTorch profile. Pass the smallest project extras and groups that the job
imports. For example, the Python API smoke job uses:

```bash
uv sync --locked --no-default-groups --extra cpu --extra yolo --group test
```

The docs job uses:

```bash
uv sync --locked --no-default-groups --extra cpu --group docs
```

CUDA jobs should replace `--extra cpu` with `--extra cu130`; the two profiles
are mutually exclusive.

CI invokes `.venv/bin` commands directly after syncing because uv does not
persist an activated optional extra. A later plain `uv run` could otherwise
re-sync without the selected CPU/CUDA profile.

## Typical CI-sensitive changes

- adding a new tracker
- renaming tracker identifiers
- changing experiment IDs used by benchmark jobs
- modifying default tracker sets used in benchmark tables or matrices
- changing ReID, mask, OBB, or native-backend requirements

## Keep docs and CI aligned

If a tracker is exposed in the docs as supported, make sure the relevant tests
and workflow coverage reflect that support level. In particular, inspect the
`TRACKERS`, `REID_TRACKERS`, `EXPECTED_OBB_TRACKERS`, and `CPP_TRACKERS`
environment lists in `ci.yml`, plus the explicit tracker/backend matrix in
`benchmark.yml`. Mask-aware trackers may need a dedicated mask source or model
instead of the generic bounding-box smoke command.
