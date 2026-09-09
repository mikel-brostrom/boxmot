# CI and Benchmarks

When a change affects benchmarked trackers or supported tracker lists, check the workflow matrices under `.github/workflows`.

## Workflow triggers

The top-level `on:` block decides whether any job in a workflow is created:

| Workflow | Current triggers |
| --- | --- |
| `.github/workflows/ci.yml` | Pushes to `master` and pull requests targeting `master` |
| `.github/workflows/docs.yml` | Pushes to `master`, pull requests targeting `master`, and manual dispatch; deployment from `master` |
| `.github/workflows/benchmark.yml` | Manual dispatch only |
| `.github/workflows/wheels.yml` | Reusable release gate and manual dispatch |
| `.github/workflows/publish.yml` | Manual dispatch after the wheel and Docker gates pass |
| `.github/workflows/docker.yml` | Reusable pre-publication gate and published releases |

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
uv sync --locked --no-default-groups --extra cpu --group docs --group test
```

CUDA jobs should replace `--extra cpu` with `--extra cu130`; the two profiles
are mutually exclusive.

CI invokes `.venv/bin` commands directly after syncing because uv does not
persist an activated optional extra. A later plain `uv run` could otherwise
re-sync without the selected CPU/CUDA profile.

Ubuntu jobs install system dependencies through
`.github/actions/install-ubuntu-packages`. Both APT update and installation use
the runner's Ubuntu source file, so unrelated vendor repositories such as
Google Chrome cannot block native builds or metric reporting. Package hash and
signature checks remain enabled. Downloads retry up to three times, and an
Ubuntu repository failure still fails the job.

The `materialize` job creates the MOT17-mini detection and embedding build once
on Python 3.12. An exact-key Actions cache reuses that build and its resolved
model artifacts on later runs only when the lockfile, package sources, dataset
fixture, workflow, and asset preparation inputs still match. The canonical
materialize command validates a restored build without rerunning inference. The
job then uploads a run-scoped bundle that both Python-version matrices of
`tune` and `metrics` download and validate. This avoids four independent
inference passes and prevents concurrent jobs from racing to populate the same
cache key.

## Typical CI-sensitive changes

- adding a new tracker
- renaming tracker identifiers
- changing experiment YAML filenames used by benchmark jobs
- modifying default tracker sets used in benchmark tables or matrices
- changing ReID, mask, OBB, or native-backend requirements

## Keep docs and CI aligned

If a tracker is exposed in the docs as supported, make sure the relevant tests
and workflow coverage reflect that support level. In particular, inspect the
`TRACKERS`, `REID_TRACKERS`, `EXPECTED_OBB_TRACKERS`, and `CPP_TRACKERS`
environment lists in `ci.yml`, plus the explicit tracker/backend matrix in
`benchmark.yml`. Mask-aware trackers may need a dedicated mask source or model
instead of the generic bounding-box smoke command.

## Release gates

Start the **Publish to PyPI** workflow on the current branch revision and select
`patch`, `minor`, or `major` under **Version bump type**. The default is `patch`.
For example, `24.0.0` becomes `24.0.1`, `24.1.0`, or `25.0.0`, respectively.

Preparation first verifies that `pyproject.toml`, `boxmot/__init__.py`, and the
local editable `boxmot` record in `uv.lock` agree. It uses
`uv version --bump <type> --no-sync` to update the project and lockfile, then
synchronizes the runtime version. The resulting candidate commit is retained
in the `release-source` bundle artifact without pushing the branch or a tag.

Every release gate restores that exact candidate. The reusable wheel workflow
runs the full tests, strict documentation build, native checks, and clean-wheel
imports on Python 3.10 through 3.13. Docker validates `cli-cpu`, `cli-gpu`, and
`service-cpu` by default, without pushing them. Package checks compare the
requested release version with the candidate and installed artifacts; they do
not pin a particular version.

After these gates pass, the workflow publishes the checksummed wheel and source
distribution. Successful **PyPI** publication is followed by an atomic push of
the tested version commit and its `v<version>` tag, then a GitHub release at that
commit. The release event starts Docker image publication. **TestPyPI** runs
the same preparation and validation but leaves the remote branch, tags, and
GitHub releases unchanged.

Release runs targeting the same package repository are serialized. The workflow
checks that the branch still points at the selected revision before preparation
and immediately before publication. It rejects an existing release tag before
PyPI publication. The final branch push uses a normal fast-forward update. If
publication succeeds but that push fails,
recover the tested candidate from the bundle artifact instead of rebuilding or
force-pushing another commit. PyPI publication requires `RELEASE_PAT` with
permission to push the version commit and tag and create the release.

The `service-gpu` target is disabled by default so releases can run without a
GPU runner. To enable its build, smoke test, and image publication, set the
repository Actions variable `BOXMOT_GPU_SERVICE_CI` to `true` after registering
a `gpu-latest` Linux NVIDIA runner with Docker and the NVIDIA Container Toolkit.
This setting applies to release gates, published releases, and manual Docker runs.
The enabled smoke exposes the GPU to the container, performs CUDA-backed ReID
enrichment through the HTTP service, and verifies that the service owns an
active CUDA context. The CPU
service smoke uses the CPU-only Torch image and exercises the same `/v1`
request boundary without CUDA.
