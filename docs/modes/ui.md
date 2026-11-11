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

Settings are arranged in stages, with only the active stage's fields visible.
For tracking, follow **Setup → Perception → Tracking → Output**:

| Stage | Settings |
| --- | --- |
| Setup | Command, source, device, image size, and half precision |
| Perception | Detector, ReID, segmentor, confidence threshold, and class filters |
| Tracking | Tracker and backend |
| Output | Display, saved results, and review before running |

The stages adapt to the command and tracker. Empty or inapplicable stages
disappear; other commands use relevant model, training, evaluation, search,
or export stages. You can revisit any stage without losing your edits.

Tracking geometry follows the detector automatically. **Details** and review
show **Automatic from detector**: `yolov8n` produces AABB tracking, while an OBB
detector produces OBB tracking. There is no separate geometry selector.
Image size controls inference resolution and does not change the box format.

On wide terminals, the **Details** panel follows the selected field with its
help text, current value, CLI default, and editing action. Select **Experiment**
to inspect its resolved settings, which also appear during review. Narrow
terminals keep field help below the form.

Press Right on **Command** or a field with choices, such as **Detector**, to
open its selector. On wide terminals, choices replace **Details** while the
form stays visible; narrow terminals show a full-width selector. Enter applies
the choice, and Left or Esc closes the selector without applying changes.

The form exposes a subset of each command's options. Use the direct CLI for
advanced workflows such as time-variant materialization, sensor class profiles,
EdgeTAM guidance, and KF/CMC selection. `boxmot MODE --help` lists every option;
settings absent from the launcher can be added to a copied command before
running it from the shell.

Defaults come from the existing CLI options. Selecting an experiment uses its
authored detector, ReID profile, and dataset settings. For `eval` and `tune`,
clear the experiment before choosing dataset and component selectors directly;
the normal CLI rules for matching a catalog experiment still apply.

Selecting `cpp` for the tracker backend can compile its native library on first
use. Install the [native build dependencies](../native/index.md#install-dependencies)
before running it. Select `python` to use the Python tracker backend.

| Key | Action |
| --- | --- |
| Up / Down | Select a field, continuing into adjacent stages |
| Tab / Shift-Tab | Select the next / previous stage |
| `1`–`9` in the main form | Jump to a stage |
| Home | Return to Command in Setup |
| End | Select Review and run in the final stage |
| Right | Open choices for Command or a field with a catalog or fixed options |
| Enter | Edit a value, toggle a flag, or open a selector |
| Enter in a selector | Apply the selected value |
| Type in a selector | Search its catalog or enter a custom value |
| Space in a multiple-choice selector | Toggle the highlighted choice |
| Ctrl-U in a selector | Clear the search or value |
| Left in a choice selector | Close without applying changes |
| `d` | Restore the selected field's CLI default |
| `r` | Review the command |
| Esc | Cancel editing, close review, or exit from the main form |
| `q` in the main form | Exit without running |

Review the command before choosing **Run**. The launcher invokes the same
command as the CLI, so downloads, model loading, and output generation happen
only after you run it. Use the command preview to repeat a run from a shell.
On the review screen, use Up / Down or Page Up / Page Down to scroll the full
command and resolved settings, then press Enter to run.

## Track selected objects with EdgeTAM

Choose **Track a video or camera**, set the source, and select tracker
`edgetam`. The **Tracking** stage shows **Prompt weights**, and the
**Perception** stage disappears because this tracker uses interactive prompts.
Geometry is shown as **AABB (prompts)**.
After **Run**, a local camera shows a live preview. Press Space or Enter when
you see the frame you want to annotate; that exact frame freezes for prompts.
Video files pause on their first frame. The window header shows **Live preview**,
**Paused**, or **Tracking**. Click the toolbar controls or use their displayed
keyboard shortcuts:

1. Select **Box** (`b`) and drag a bounding box, or select **Points** (`p`) and
   left-click to add green `+` include points. Right-click adds red `-` exclude
   points with either tool, including directly after drawing a box for the
   same object. The current tool stays selected.
2. Select **New object** (`n`) to add another object's prompt.
3. Use the playback control or press Space or Enter to start tracking. Use it
   again to pause on the next frame and add or correct prompts.

The point legend stays visible in the window. New or edited dots also appear
on that frame's tracking result. When paused, the selected object's saved
points appear as hollow dots marked **prev** at their last clicked positions.
Their positions stay fixed as the video changes.

While paused, click an object chip or use Tab to select an existing object;
the selected chip is highlighted. Right-click unwanted areas, then click
**Start tracking**. Existing objects can be corrected with exclude points
alone, using their mask propagated to this frame. A new object needs a box or
at least one include point.

Click a dot to select it and drag to move it. Press `d` to delete it, `i` to
make it an include point, or `e` to make it an exclude point. New, moved, or
relabeled points apply to the current frame when you resume; unchanged saved
points are not resubmitted. Deleting a previously submitted marker does not
undo its earlier effect on the mask. To restore an area, add an include point
on the current frame or draw a fresh box.

The toolbar also provides Undo (`u`), Clear (`c`), and Remove (`x`). Clear
discards this frame's draft edits and restores saved markers; Undo reverses
the last draft edit. Remove stops tracking the selected object and clears its
points. A new sequence clears all saved points. Use `q` or Esc to stop. While
paused on a camera, press `r` to discard draft edits and return to live preview
to choose another frame.

Camera capture continues in the background during annotation and inference,
keeping only the newest frame for the next update. Your selected prompt frame
stays frozen until it is processed. This reduces camera backlog; model speed
still limits tracking and display, especially on CPU. Video files remain
sequential. Saved video includes propagated masks and IDs only for processed
frames; preview frames and skipped camera captures are not saved. This requires
a desktop display and the optional EdgeTAM dependencies; see
[EdgeTAM setup](../trackers/edgetam.md).

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
