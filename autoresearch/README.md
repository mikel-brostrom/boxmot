# autoresearch

`autoresearch/` is a BoxMOT-focused experiment scaffold for autonomous tracker work.

Run everything from the repo root with the main BoxMOT environment. The research loop here is not a standalone training project anymore; it is a thin wrapper around:

- `boxmot/engine/evaluator.py`
- `boxmot/engine/tuner.py`

## What it does

The original Karpathy-style idea still applies: let an agent iterate autonomously, but instead of mutating one GPT training file, the agent edits BoxMOT tracker code and validates changes against benchmark metrics.

This version supports any tracker that has a runtime YAML in `boxmot/configs/trackers/`.

Typical edit targets are:

- `boxmot/trackers/<tracker>/...`
- `boxmot/configs/trackers/<tracker>.yaml`
- tracker-specific tests and docs when behavior changes

## Files that matter

- `prepare.py`
  warms detection/embedding caches through `boxmot.engine.evaluator.run_generate_dets_embs(...)` and initializes `results.tsv`
- `train.py`
  runs one `eval` or `tune` experiment by calling BoxMOT's existing evaluator/tuner entry points
- `program.md`
  autonomous-agent instructions for the experiment loop

## Harness vs Edit Surface

This fork keeps the original three-file autoresearch structure, but the roles are different from the upstream GPT-training example.

- `prepare.py` is setup code. The agent should not normally modify it during a research run.
- `train.py` is the experiment runner. It is the metric oracle wrapper around `evaluator.py` and `tuner.py`. It does not modify tracker code by itself.
- `program.md` is the human-authored research policy for the agent.

The actual code-under-test lives in BoxMOT:

- `boxmot/trackers/<tracker>/...`
- `boxmot/configs/trackers/<tracker>.yaml`
- tracker-specific tests and docs when behavior changes

In other words: the agent edits tracker implementations, then uses `python -m autoresearch.train ...` to measure whether those edits improved benchmark metrics.

## Why this is useful

Most tracker iterations do not change the detector or benchmark data. Because `run_generate_dets_embs(...)` caches detections and embeddings, the expensive detector pass can be reused across code changes. That keeps the research loop focused on:

- tracker logic changes
- tracker YAML default/search-space changes
- benchmark evaluation and tuning

By default, autoresearch keeps its JSON artifacts under `runs/autoresearch/...` but leaves BoxMOT runtime outputs in the normal `runs/` tree. That means detections and embeddings are reused from `runs/dets_n_embs/...` instead of being regenerated under an autoresearch-specific folder. If you need a different BoxMOT runtime root, pass `--runtime-project`.

The intended loop is:

1. Edit tracker code or tracker YAML.
2. Run `autoresearch.train eval` to score the current code.
3. Optionally run `autoresearch.train tune` to search better defaults for that tracker.
4. Keep or discard the code change based on the resulting metrics.

## Quick start

Warm the benchmark caches and initialize the result ledger:

```bash
uv run python -m autoresearch.prepare \
  --benchmark mot17-ablation \
  --tracker bytetrack
```

Run a baseline evaluation:

```bash
uv run python -m autoresearch.train eval \
  --benchmark mot17-ablation \
  --tracker bytetrack \
  --record \
  --status keep \
  --description "baseline"
```

Run a tuning sweep against the tracker's YAML search space:

```bash
uv run python -m autoresearch.train tune \
  --benchmark mot17-ablation \
  --tracker bytetrack \
  --n-trials 12 \
  --objectives HOTA \
  --record \
  --status keep \
  --description "baseline sweep"
```

Swap `bytetrack` for any other supported tracker such as `botsort`, `deepocsort`, `strongsort`, `ocsort`, `hybridsort`, `boosttrack`, or `sfsort`.

## Output artifacts

The helpers write JSON artifacts under `runs/autoresearch/<tracker>/<benchmark>/` by default:

- `setup.json`
- `last_eval.json`
- `last_tune.json`

The underlying BoxMOT runtime outputs still go to the standard project root by default:

- detections and embeddings: `runs/dets_n_embs/<benchmark>/...`
- MOT result files: `runs/mot/<benchmark>/...`
- tune results: `runs/ray/...`

They also optionally append experiment rows to `autoresearch/results.tsv`.

The ledger schema is:

```text
commit	tracker	benchmark	phase	HOTA	MOTA	IDF1	AssA	AssRe	IDSW	IDs	IDSW_rate	status	description
```

- `commit` is the 7-character experiment commit hash.
- `tracker` is the tracker under test.
- `benchmark` is the benchmark bundle that was evaluated.
- `phase` is the experiment type, such as `eval` or `tune`.
- `HOTA`, `MOTA`, and `IDF1` are the primary comparison metrics.
- `AssA`, `AssRe`, `IDSW`, `IDs`, and `IDSW_rate` are secondary metrics and guardrails.
- `status` is `keep`, `discard`, or `crash`.
- `description` is a short summary of the idea that was tested.

Use `uv run python -m autoresearch.log` to append a result row after you decide whether an experiment should be kept or discarded. The ledger is tab-separated and is intended to stay untracked by git.

If you already have an older `results.tsv` with a different schema, the next write will migrate it automatically by renaming the old file to `results.legacy.tsv` (or `results.legacy.<n>.tsv`) and starting a fresh ledger.

## Notes

- Motion-only trackers still pass through the standard evaluator/tuner interfaces. Keep the `--reid` input aligned with your benchmark workflow even if the tracker ignores appearance features.
- Use the wrappers here for the stable outer loop, but inspect `boxmot/engine/evaluator.py` and `boxmot/engine/tuner.py` directly when changing the research process itself.
- `autoresearch.train` does not edit source files. The LLM agent edits tracker files; `autoresearch.train` evaluates the current working tree.
