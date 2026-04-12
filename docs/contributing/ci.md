# CI and Benchmarks

When a change affects benchmarked trackers or supported tracker lists, check the workflow matrices under `.github/workflows`.

## Typical CI-sensitive changes

- adding a new tracker
- renaming tracker identifiers
- changing benchmark names
- modifying default tracker sets used in benchmark tables or matrices

## Keep docs and CI aligned

If a tracker is exposed in the docs as supported, make sure the relevant tests and workflow coverage reflect that support level.
