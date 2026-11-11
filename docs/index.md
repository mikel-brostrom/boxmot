# BoxMOT

BoxMOT is a multi-object tracking toolkit for axis-aligned boxes, oriented
boxes, and instance masks. Track videos, image sequences, and live streams
from the CLI, or use its detectors, appearance encoders, and trackers in Python.

Swap components to suit your footage, reuse detections and embeddings across
experiments, and evaluate or tune trackers on the same inputs. Selected trackers
also provide native C++ backends. The [tracker overview](trackers/index.md)
lists supported inputs, including mask-aware tracking and 2D/3D sensor fusion.

## Get started

```bash
pip install boxmot "ultralytics>=8.4.58"
boxmot --help
```

Choose **CLI** or **Python** below. The docs remember your choice across
examples with the same tab labels.

=== "CLI"

    ```bash
    boxmot track \
      --source video.mp4 \
      --detector yolov8n \
      --tracker bytetrack \
      --save
    ```

=== "Python"

    ```python
    from boxmot.engine.tracking.workflow import run_track

    result = run_track(
        source="video.mp4", detector="yolov8n", tracker="bytetrack", save=True
    )
    print(result.video_path)
    ```

Replace `video.mp4` with your video. The detector weights download on first use;
`--save` writes the tracked video. See [Track](modes/track.md) for webcams,
streams, and display options, or [Installation](getting-started/installation.md)
for other backends.

## Compare trackers on a dataset

Materialize detections and appearance embeddings once, then reuse that build
for evaluation. This YOLOX example uses the
[source checkout with the `yolo` extra](getting-started/pytorch.md#source-checkout):

=== "CLI"

    ```bash
    boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
    boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
    ```

=== "Python"

    ```python
    from boxmot.engine.eval.evaluator import run_eval
    from boxmot.engine.materialization.workflow import materialize

    if __name__ == "__main__":
        build = materialize(experiment="mot17/ablation-yolox-lmbn.yaml")
        result = run_eval(experiment="mot17/ablation-yolox-lmbn.yaml", build=build)
        print(result.summary)
    ```

Materialization downloads the configured MOT17 split and model weights as
needed. Replace `BUILD_ID` with the ID it prints. See
[supported datasets](datasets/index.md) to choose a benchmark and
[comparison guidance](guides/comparison.md) to keep experiments comparable.
The Python example passes the returned build path directly to evaluation.

## Where to go next

| Goal | Guide |
| --- | --- |
| Choose a tracker and check its input requirements | [Trackers](trackers/index.md) |
| Choose box, oriented-box, mask, or sensor-fusion tracking and its variants | [Tracking tasks](tasks/index.md) |
| Choose a benchmark or bring your own data | [Datasets](datasets/index.md) |
| Materialize, evaluate, tune, or train ReID models | [Modes](modes/index.md) |
| Compose components and pipelines in Python | [Python API](python/index.md) |
| Deploy with Docker, HTTP, or native C++ | [Integrations](integrations/index.md) |

For component boundaries and data contracts, see the
[tracking pipeline](concepts/tracking-pipeline.md) and
[canonical structures](concepts/index.md) guides.
