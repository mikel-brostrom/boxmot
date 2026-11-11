# EdgeTAM

Select objects with bounding boxes or points and track their masks through a
video using [EdgeTAM](https://github.com/facebookresearch/EdgeTAM). BoxMOT
maintains a separate model memory for each selected object and returns masks,
enclosing AABBs, and stable IDs. No detector or ReID encoder is needed.

Select this workflow with `--tracker edgetam`. The separate `--edgetam` flag
enables [temporal mask guidance](../tasks/masks.md#use-temporal-masks-in-association)
for detector-driven box trackers such as ByteTrack; it does not select this
prompted tracker.

## Setup and UI

From a BoxMOT source checkout, install the optional backend and launch the
existing terminal menu:

```bash
uv sync --extra cpu --extra yolo --group edgetam
uv run --no-sync boxmot ui --command track
```

Use `--extra cu130` instead of `--extra cpu` for the CUDA 13.0 profile. Select
tracker **edgetam**, enter your video path or camera index as **Source**, and
choose the inference device. **Prompt weights** defaults to `edgetam.pt`, which
downloads on the first prompted frame if missing; a local checkpoint path is
also accepted. A desktop display and OpenCV with GUI support are required.

The backend builds once from the pinned source archive. BoxMOT configures its
build to use the selected CPU or CUDA PyTorch version. If an installation
started before this configuration was updated is still showing **Building
edgetam**, stop it with Ctrl+C and rerun the sync command. For a CPU-only
installation, explicitly skip upstream's optional CUDA extension and show
build progress with:

```bash
SAM2_BUILD_CUDA=0 uv sync --extra cpu --extra yolo --group edgetam -v
```

After sync finishes, launch the UI with `uv run --no-sync boxmot ui --command
track`. Later UI launches do not need another sync unless dependencies change.

Review the command and choose **Run**. A local camera opens a live preview.
Wait until the picture looks right, then press Space or Enter to freeze the
displayed frame. Draw boxes or points on it, then press Space or Enter again
to start tracking. Press `r` while paused to discard the draft edits and choose
another camera frame. The camera preview has no fixed warmup delay and does
not run the model or save frames until tracking starts.

For local cameras opened by this workflow, capture continues in the background
during inference and annotation, keeping only the latest frame. The chosen
prompt frame stays frozen until it is processed; the next update uses the
newest camera frame and skips older captures. This reduces camera backlog,
but tracking and display speed still depend on model inference, which can be
slow on CPU. Video files pause on their first frame for prompts and then
process frames sequentially. You can launch the same window directly:

```bash
boxmot track --tracker edgetam --source video.mp4 --prompt-weights edgetam.pt --save
```

The window header shows **Live preview**, **Paused**, or **Tracking**. Click
the toolbar controls or use their displayed keyboard shortcuts. While paused,
choose **Box** or **Points** and click an object chip to select it for editing;
the selected chip is highlighted.

| Control in the video window | Action |
| --- | --- |
| `b`, then left-drag | Draw a bounding box for the selected object |
| `p`, then left-click | Add a green `+` point to include foreground |
| Right-click | Add a red `-` point to exclude background |
| Click / drag a dot | Select / move a point |
| `d` | Delete the selected point |
| `i` / `e` | Change the selected point to include / exclude |
| `n` | Select a new object |
| Tab | Select an existing object to correct |
| `u` / `c` | Undo a draft edit / discard this frame's edits and restore saved points |
| `x` | Remove the selected object |
| Space / Enter | Choose the displayed camera frame; after prompting, start or pause tracking |
| `r` while paused on a camera | Discard draft edits and choose another frame from live preview |
| `q` / Esc | Stop and close the video window |

After drawing a box, right-click to add exclude points for the same object.
Right-click works with either **Box** or **Points** and keeps the current tool
selected. Use **Points** (`p`) to add green `+` include points with left-click.

The window keeps a legend for both point types visible. New or edited dots
also appear on that frame's tracking result. When paused, the selected
object's saved points reappear as hollow dots marked **prev** at their last
clicked positions. These positions stay fixed as the video changes.

Click a dot to select it, then drag it, delete it with `d`, or change it to
include (`i`) or exclude (`e`). Resume to apply new, moved, or relabeled points
to the current frame. Selecting an object or resuming without edits does not
resubmit its saved points.

Deleting a previously submitted point removes its marker but does not undo
its earlier effect on the mask. To restore an excluded area, add an include
point on the current frame or draw a fresh box. Clear (`c`) discards draft
edits and restores saved markers. Removing an object clears its points;
starting a new sequence clears all saved points.

To refine an existing tracked object with exclude points, pause, select its
chip, right-click the areas to exclude, then click **Start tracking**. The
correction uses the object's mask propagated to the selected frame, so it does
not need a fresh box or include point. New objects require a box or at least
one green `+` include point.

Prompts apply to the displayed frame. Corrections affect only the selected
object. New sequences start with fresh object IDs; local
cameras return to live frame selection and files pause for prompts.
At most 32 objects are retained by default; `EdgeTAMConfig.max_objects` controls
the limit and `batch_size` controls propagation batching.

## Python prompts

The tracker is also usable without a GUI. Here `bgr_frame` is a decoded
`uint8[H,W,3]` image; use a new frame on each update:

```python
import numpy as np

from boxmot import EdgeTAM, EdgeTAMModelConfig

tracker = EdgeTAM(prompt_model=EdgeTAMModelConfig(checkpoint="edgetam.pt", device="cpu"))
person = tracker.add_box((20, 30, 180, 260))
ball = tracker.add_points([(240, 120)], labels=[1])
rows = tracker.update(np.empty((0, 6), dtype=np.float32), bgr_frame)
```

Use `add_box(..., object_id=person)` or `add_points(..., object_id=person)`
before an update to correct that object on the incoming frame. Point labels
are `1` for foreground and `0` for background. Use `remove_object(id)` to stop
tracking one object and `reset()` to start a new sequence.
Point-only prompts for new objects need at least one foreground point.
Existing tracked objects also accept background-only points: pass their
`object_id` to `add_points` to refine the mask propagated to the incoming frame.
If that mask is empty, a foreground point or replacement box may be needed to
recover the object. To refine a pending box with points, call `add_box` and
then `add_points` with the same object ID before the next update.

Canonical empty `Detections` plus a `Frame` return `Tracks` with full-frame
boolean masks. Packed NumPy input returns eight-column AABB rows. Detection
indices are `-1`, classes default to `0`, and output scores are `1.0` rather
than calibrated detector confidence. Empty masks produce no visible row while
the object's model memory is retained.

`--save` writes annotated video. `--save-txt` and `--save-json` use the normal
[tracking outputs](../modes/track.md#save-or-display-results), which store boxes
and IDs; they do not serialize masks. Saved outputs contain only processed
frames; skipped camera captures are omitted. Source frame identities and
timestamps are preserved where available, including gaps from skipped captures.

Detector-driven evaluation and tuning, OBB, per-class tracking, and the native
C++ backend are not supported for this prompted tracker.

## Evaluate first visible people

Evaluate EdgeTAM on MOTChallenge-style pedestrian ground truth by initializing
each identity from its first fully visible, fully in-frame annotation:

```bash
boxmot eval --dataset mot17-mini --split train --tracker edgetam \
  --prompt-source first-visible --prompt-weights edgetam.pt --device cpu
```

Install the EdgeTAM backend using the setup command above first. `--data-root`
selects your dataset location; `--sequence NAME` limits the run to one sequence.
The workflow reads images and ground truth directly, without a perception build.
It uses the Python backend and one sequence worker. Add `--save` for annotated
videos or `--show` for a live preview.

The default `--prompt-min-visibility 1.0` requires a marked pedestrian with
visibility 1.0 and an entire bounding box inside the actual image dimensions.
A partially visible first appearance is skipped until an eligible annotation
appears. Lower the threshold explicitly, for example
`--prompt-min-visibility 0.95`, to permit some occlusion; the entire box must
still be in frame. Ground truth must include its visibility column.

Each identity receives one box prompt per sequence. Later annotations never
correct the tracker, remove an object, or reinitialize an identity after it
disappears. `max_objects` therefore needs to cover every prompted identity in
the sequence, including people no longer visible. To increase the default 32,
save a tracker configuration such as `edgetam-eval.yaml`:

```yaml
max_objects: 128
batch_size: 8
```

Then add `--tracker-config edgetam-eval.yaml`. Evaluation checks capacity before
loading the model; increase it further if the selected sequence requires it.

HOTA, MOTA, and IDF1 score the complete sequence against the original ground
truth. Frames before a person's prompt and people who never qualify still
count as missed ground truth. These results use ground-truth initialization
and measure a different protocol from detector-driven tracking.

The run directory contains MOT track text files, `metrics.json`,
`eval_manifest.json`, and `prompts.json` recording each initialization.
See [evaluation](../modes/eval.md#edgetam-with-ground-truth-prompts) for the
supported command options.

## Tracker API

::: boxmot.trackers.edgetam.tracker.EdgeTAM
    options:
      show_root_heading: false
