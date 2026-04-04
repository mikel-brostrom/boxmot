# autoresearch

This is an autonomous research loop for BoxMOT trackers.

The harness is tracker-agnostic. Pick a tracker such as `bytetrack`, `botsort`, `deepocsort`, `strongsort`, `ocsort`, `hybridsort`, `boosttrack`, or `sfsort`, then evolve its code and/or YAML config against benchmark metrics.

Important role split:

- `autoresearch/prepare.py` is setup code.
- `autoresearch/train.py` is the experiment runner.
- The agent's actual edit targets are the tracker implementation, its YAML defaults/search space, and relevant tests/docs.

Do not treat `autoresearch/train.py` as the algorithm-under-test. Treat it as the stable evaluation harness around `boxmot/engine/evaluator.py` and `boxmot/engine/tuner.py`.

## Setup

Work with the user to establish:

1. A tracker name.
2. A benchmark bundle.
3. A run tag.

Then:

1. Create a dedicated branch such as `autoresearch-<tracker>-<tag>`.
2. Read the in-scope files:
   - `autoresearch/README.md`
   - `autoresearch/program.md`
   - `autoresearch/prepare.py`
   - `autoresearch/train.py`
   - `boxmot/engine/evaluator.py`
   - `boxmot/engine/tuner.py`
   - `boxmot/configs/trackers/<tracker>.yaml`
   - `boxmot/trackers/<tracker>/...`
   - the most relevant tracker tests, usually under `tests/unit/`
3. Warm caches and initialize the ledger:

   ```bash
   uv run python -m autoresearch.prepare \
     --benchmark <benchmark> \
     --tracker <tracker>
   ```

   By default this reuses BoxMOT caches under `runs/dets_n_embs/<benchmark>/...`. Only set `--runtime-project` if you intentionally want a different BoxMOT runtime root.

4. Establish a baseline evaluation:

   ```bash
   uv run python -m autoresearch.train eval \
     --benchmark <benchmark> \
     --tracker <tracker> \
     --record \
     --status keep \
     --description "baseline"
   ```

5. If the tracker has a meaningful YAML search space, also establish a baseline sweep:

   ```bash
   uv run python -m autoresearch.train tune \
     --benchmark <benchmark> \
     --tracker <tracker> \
     --n-trials 12 \
     --objectives HOTA \
     --record \
     --status keep \
     --description "baseline sweep"
   ```

Once setup is complete, start iterating without waiting for more confirmation.

## What to change

Primary edit targets:

- `boxmot/trackers/<tracker>/...`
- `boxmot/configs/trackers/<tracker>.yaml`

Default assumption:

- tracker YAML defaults are already tuned enough that single-parameter nudges are usually low value
- do not spend the research budget on one-knob sweeps such as changing only `match_thresh`, only `track_buffer`, or only one confidence cutoff
- use YAML edits mainly when they are required to support or expose a real algorithmic change

Preferred change classes:

- these are preferred high-value examples, not an exclusive list
- association-level changes in the first-pass, second-pass, unconfirmed-track, or lost-track matching logic
- score fusion changes, cost-matrix construction changes, gating changes, and match ordering changes
- embedding-matching changes for trackers that use appearance cues, including how appearance distance is combined with motion or IoU cues
- track lifecycle changes that are tightly coupled to association behavior, such as confirmation, re-activation, or duplicate suppression rules

Secondary edit targets, only when necessary:

- shared utilities used directly by that tracker
- tests that need to reflect new behavior
- docs that mention tracker defaults or capabilities

## What to use

Prefer the existing benchmark pipeline over ad hoc scripts:

- use `boxmot/engine/evaluator.py` through `python -m autoresearch.train eval`
- use `boxmot/engine/tuner.py` through `python -m autoresearch.train tune`

Do not invent a parallel evaluation harness when the existing one already measures the correct metrics.
Do not expect `autoresearch.train` to rewrite tracker code for you; it evaluates whatever code is currently in the working tree.

## Optimization targets

Main primary metric:

- `HOTA`

Other primary metrics:

- `MOTA`
- `IDF1`

Secondary metrics and guardrails:

- `AssA`
- `AssRe`
- `IDSW`
- `IDs`
- `IDSW_rate` should generally not regress badly even if HOTA rises slightly

Default decision rule:

- optimize for `HOTA` first
- use `MOTA` and `IDF1` as the other primary checks
- use `AssA`, `AssRe`, `IDSW`, `IDs`, and `IDSW_rate` as secondary context and guardrails

Use judgment. A tiny metric gain that adds brittle complexity is usually not worth keeping.

Do not treat the experiment loop as a hyperparameter sweep around already tuned defaults. Prefer structural or algorithmic tracker changes that can plausibly improve overall tracking performance.

## Logging

Log completed experiments to `autoresearch/results.tsv`. The file is tab-separated, is intentionally kept out of git, and should never be staged in experiment commits.

When you compare or summarize experiments, keep the full result columns visible:

```text
commit	tracker	benchmark	phase	HOTA	MOTA	IDF1	AssA	AssRe	IDSW	IDs	IDSW_rate	status
```

- `commit`: short 7-character git commit hash for the experiment commit that was run
- `tracker`: tracker name under test
- `benchmark`: benchmark bundle that was evaluated
- `phase`: experiment type such as `eval` or `tune`
- `HOTA`: main primary metric, formatted as a decimal string with 6 digits after the decimal point
- `MOTA`: primary comparison metric
- `IDF1`: primary comparison metric
- `AssA`, `AssRe`, `IDSW`, `IDs`, and `IDSW_rate`: secondary comparison metrics and guardrails
- `status`: `keep`, `discard`, or `crash`

Use `uv run python -m autoresearch.log` to append a keep/discard/crash row after you have seen the outcome of an experiment. For one-step cases where the outcome is already known, such as the baseline, `autoresearch.train eval|tune --record --status keep ...` is fine.

If a local helper only writes a narrower ledger row, keep the full metric set from `run.log`, `last_eval.json`, or `last_tune.json` in your notes when comparing experiments.

Use short, factual descriptions such as:

- `baseline`
- `tighten second-pass matching`
- `smooth OBB angle update`
- `expand tune range for track_buffer`
- `crash: invalid shape in association path`

## Experiment loop

Repeat indefinitely:

1. Inspect the current branch and the last kept result.
2. Make one focused tracker change.
   The default should be an algorithmic or structural change aimed at general tracking performance, not a single-parameter tweak. Association and embedding-matching changes are good examples, but not the only valid category.
3. Run the most relevant tests for that tracker.
4. Stage only the code, config, and test files for this experiment. Never stage `autoresearch/results.tsv` or `run.log`.
5. Commit the experiment before you run it so the ledger can record the exact code state.

   Suggested format:

   ```bash
   git commit -m "feat: autoresearch <change summary>"
   ```

6. Run the experiment with all output redirected to `run.log`. Do not use `tee`, and do not let the benchmark output flood the agent context.

   ```bash
   uv run python -m autoresearch.train eval \
     --benchmark <benchmark> \
     --tracker <tracker> \
     > run.log 2>&1
   ```

7. Read the result from `run.log`:

   ```bash
   grep "^HOTA:\|^MOTA:\|^IDF1:\|^AssA:\|^AssRe:\|^IDSW:\|^IDs:\|^IDSW_rate:\|^summary_json:" run.log
   ```

   If the grep output is empty, treat the run as a crash. Read the traceback with:

   ```bash
   tail -n 50 run.log
   ```

8. If the run succeeded, compare it against the last kept result and decide `keep` or `discard`, with `HOTA` first, `MOTA`/`IDF1` next, and the remaining metrics used as secondary context and guardrails.
9. Log the completed experiment after the decision:

   ```bash
   uv run python -m autoresearch.log \
     --artifact runs/autoresearch/<tracker>/<benchmark>/last_eval.json \
     --status <keep|discard> \
     --description "<change summary>"
   ```

   For tuning runs, point `--artifact` at `last_tune.json`. For crashes, omit `--artifact` and log:

   ```bash
   uv run python -m autoresearch.log \
     --status crash \
     --description "<change summary>"
   ```

10. If the experiment is a keep, advance the branch by leaving the experiment commit in place and continue from it.
11. If the experiment is a discard, reset back to the pre-experiment commit only when it is safe to drop that single experiment commit without discarding unrelated work.

   In the normal one-commit-per-idea case:

   ```bash
   git reset --hard HEAD^
   ```

   Do not reset away unrelated human work.

If the human gives a hard cap such as "run 10 iterations", interpret that as 10 code-change experiments, not 10 Ray Tune trials. Count one iteration each time you:

1. make a focused tracker/code/config change, and
2. validate it with `autoresearch.train eval` or `autoresearch.train tune`.

Parameter-only edits do not count as good default experiments unless they are part of validating a broader algorithmic change.

## Crash policy

If a run crashes:

1. Inspect the traceback.
2. Fix obvious implementation mistakes and retry.
3. If the idea is fundamentally broken, log it as a crash and reset the experiment commit.

## Timeouts

Each experiment should usually finish in a few minutes once caches are warm. If a run exceeds 10 minutes, kill it, log it as a crash, and reset the experiment commit.

## Persistence

Do not stop to ask if you should continue once the loop has started. The expectation is uninterrupted autonomous iteration until the user explicitly interrupts the run.
