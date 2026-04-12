# Results and Artifacts

BoxMOT writes run outputs under the configured `project/name` directory.

## Common artifacts

Depending on workflow and flags, you may see:

- rendered video outputs
- MOT-style `.txt` tracking files
- cropped detections
- benchmark summaries
- tuning trial artifacts
- GEPA state and candidate directories
- exported ReID model files

## Tracking outputs

`track` can write:

- video or image outputs with overlays
- MOT text files via `--save-txt`
- crops via `--save-crop`

The public Python API also exposes structured result objects such as `TrackRunResult`, `Results`, and `Tracks`. See [Results Objects](../python/results.md).

## Benchmark outputs

`generate`, `eval`, `tune`, and `research` also create reusable caches and evaluation artifacts tied to the selected benchmark.

## Reproducibility tip

For repeated experiments, keep the same benchmark, detector, ReID, and tracker selections together under one project tree so cached detections and embeddings are easy to reuse.
