# Terminal UI

`boxmot ui` opens a Rich terminal launcher for common tracking, benchmarking,
and ReID commands. Choose settings, review the generated command, then run it
with an interactive progress dashboard.

=== "CLI"

    ```bash
    boxmot ui
    ```

=== "Python"

    ```python
    from boxmot.engine.cli import boxmot

    boxmot.main(["ui"], standalone_mode=False)
    ```

The Python example invokes the same interactive launcher through the CLI
command group. Run it in an interactive terminal.

Start with a particular command and experiment:

```bash
boxmot ui --command eval --experiment mot17/ablation-yolox-lmbn.yaml
```

## Select a command and settings

The launcher supports `track`, `materialize`, `eval`, `tune`, `export`,
`train-reid`, and `eval-reid`. It shows the common options for the selected
command, including model, tracker, input, device, and output settings where
applicable. Catalog selectors support searching, and path fields accept your
own files.

The form exposes a subset of each command's options. Use the direct CLI for
advanced workflows such as time-variant materialization, sensor class profiles,
EdgeTAM guidance, and KF/CMC selection. `boxmot MODE --help` lists every option;
settings absent from the launcher can be added to a copied command before
running it from the shell.

Defaults come from the existing CLI options. Selecting an experiment uses its
authored detector, ReID profile, and dataset settings. For `eval` and `tune`,
clear the experiment before choosing dataset and component selectors directly;
the normal CLI rules for matching a catalog experiment still apply.

| Key | Action |
| --- | --- |
| Up / Down / Tab | Select a field |
| Enter | Edit a value, toggle a flag, or open a selector |
| Type in a selector | Search its catalog or enter a custom value |
| Space in a multiple-choice selector | Toggle the highlighted choice |
| Ctrl-U in a selector | Clear the search or value |
| `d` | Restore the selected field's CLI default |
| `r` | Review the command |
| Esc | Close the selector or review; exit from the main form |
| `q` in the main form | Exit without running |

Review the command before choosing **Run**. The launcher invokes the same
command as the CLI, so downloads, model loading, and output generation happen
only after you run it. Use the command preview to repeat a run from a shell.
On the review screen, use Up / Down or Page Up / Page Down to scroll the full
command and resolved settings, then press Enter to run.

## Live dashboard

The launcher enables the dashboard for its commands. You can also enable it
when running a command directly by placing `--dashboard` before the command:

```bash
boxmot --dashboard eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort
```

The dashboard has **Overview**, **Sequences**, **Results**, **Config**, and
**Logs** views. Their contents depend on the command and the data it has
reported; metrics appear after scoring completes. Sequence rows expose frame
counts and status. Tuning results include observed completed trials, a
best-objective sparkline, and metric differences from trial 1 when available.
Commands without a workflow progress reporter keep their existing output.

| Key | Action |
| --- | --- |
| Tab / Shift-Tab / Left / Right | Switch views |
| `1`–`5` | Jump to a view |
| Up / Down | Select a row or scroll details |
| Enter | Inspect the selected sequence, trial, or completed step |
| Page Up / Page Down / Home / End | Scroll details |
| `s` in Results | Change the trial sort metric |
| `/` in Logs | Search messages |
| `w` in Logs | Toggle warnings and errors only |
| `f` in Logs | Freeze or follow log scrolling |
| Esc | Close details or clear a search |
| `?` | Show keyboard help |
| Ctrl-C | Interrupt the command |
| `q` after completion | Close the dashboard and return to the shell |

Navigation and freezing log scrolling do not pause computation. On successful
completion, the dashboard stays open so you can inspect results. Press `q`
when finished; the final summary remains in terminal scrollback. Automatic
materialization within an evaluation or tuning command continues into replay
without waiting for input between stages.

The launcher requires an interactive terminal, such as the VS Code integrated
terminal. Direct commands still work in scripts and redirected output;
`--dashboard` falls back to the existing output when no interactive terminal
is available.

## Related pages

- [Track](track.md)
- [Materialize](materialize.md)
- [Evaluate](eval.md)
- [Tune](tune.md)
- [Export](export.md)

## CLI arguments

::: mkdocs-click
    :module: boxmot.engine.commands.ui
    :command: ui
    :depth: 0
    :style: table
    :prog_name: boxmot ui
