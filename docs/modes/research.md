# Research

`research` proposes tracker code changes and scores them against a fixed
experiment and immutable build. Prepare its dependencies with
`boxmot install --extra research`; see [Install dependencies](install.md).
Use a source checkout with the tracker implementation available and configure
the proposal provider's credential in the environment (for example,
`OPENAI_API_KEY` for an `openai/...` selector). The command passes the selected
model identifier to GEPA; model access depends on your provider account.
It requires both experiment and build inputs:

=== "CLI"

    ```bash
    boxmot research \
      --experiment mot17/ablation-yolox-lmbn.yaml \
      --build BUILD_ID \
      --tracker bytetrack \
      --proposal-model openai/gpt-5.4 \
      --max-metric-calls 24
    ```

=== "Python"

    ```python
    from boxmot.engine.research.runner import run_research

    if __name__ == "__main__":
        result = run_research(
            experiment="mot17/ablation-yolox-lmbn.yaml",
            build="BUILD_ID",
            tracker="bytetrack",
            proposal_model="openai/gpt-5.4",
            max_metric_calls=24,
        )
        print(result.best_candidate_dir)
    ```

Replace `BUILD_ID` with the ID or directory produced by
[materialization](materialize.md) for the same experiment. The Python call
returns a `ResearchResult` with baseline/best metrics and candidate paths.

The parent process validates the build before creating isolated workspaces.
Each candidate replays selected keyed sequences through the live tracker API;
candidate workspaces do not perform perception inference or mutate the build.

Run materialization separately when required:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
```

There is no implicit build creation or latest-build lookup. This makes metric
changes attributable to tracker code or parameters rather than changing
detections and embeddings.

## Search scope and scoring

By default, proposals edit the selected tracker's Python implementation module,
identified by the tracker registry. Candidate files are checked for valid syntax
and imports before evaluation. The search runs against a copied workspace;
the original checkout is not edited automatically. Source images and annotations
must remain available wherever the selected tracker or evaluator needs them.

The objective is combined HOTA minus weighted regressions from the baseline.
Defaults subtract one point for each point of IDF1 or MOTA regression; HOTA's
additional regression penalty is zero. The `--*-penalty` and `--*-tolerance`
options change those terms. `--max-metric-calls` bounds GEPA's search evaluations;
the initial baseline and final best-candidate evaluations also run outside that
search budget. `--eval-timeout` limits each benchmark evaluation.

The current workflow scores baseline, proposals, and the final candidate on the
same selected benchmark sequences. It does not create an independent validation
split. Review the code and evaluate on held-out data before adopting a candidate.

## Saved results

Results live under `<project>/research/<tracker>_<experiment-slug>/`:

| Artifact | Contents |
| --- | --- |
| `best_candidate/` | Proposed source files at their repository-relative paths. |
| `research_result.json` | Baseline and best metrics, deltas, penalty settings, and evaluated sequences. |
| `gepa/` | Search state and evaluation history. |
| `boxmot_runs/` | Benchmark evaluation outputs. |

The temporary workspace is removed after a successful run unless
`--keep-workspace` is supplied. The best candidate and summary are retained.
Research reuses the same run directory for the same tracker/experiment and
resets existing GEPA state; use a different `--project` to preserve independent
searches. Review `best_candidate/` before copying any changes into your checkout.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.research
    :command: research
    :prog_name: boxmot research
    :depth: 0
