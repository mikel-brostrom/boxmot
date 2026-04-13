<div align="center" markdown="1">

  <img width="640"
       src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif"
       alt="BoxMOT demo">
  <br>

  <a href="https://trendshift.io/repositories/13239" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13239" alt="mikel-brostrom%2Fboxmot | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"></a>

  [![CI](https://github.com/mikel-brostrom/boxmot/actions/workflows/ci.yml/badge.svg)](https://github.com/mikel-brostrom/boxmot/actions/workflows/ci.yml)
  [![PyPI version](https://badge.fury.io/py/boxmot.svg)](https://badge.fury.io/py/boxmot)
  [![downloads](https://static.pepy.tech/badge/boxmot)](https://pepy.tech/project/boxmot)
  [![license](https://img.shields.io/badge/license-AGPL%203.0-blue)](https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE)
  [![python-version](https://img.shields.io/pypi/pyversions/boxmot)](https://badge.fury.io/py/boxmot)
  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8132989.svg)](https://doi.org/10.5281/zenodo.8132989)
  [![docker pulls](https://img.shields.io/docker/pulls/boxmot/boxmot?logo=docker)](https://hub.docker.com/r/boxmot/boxmot)
  [![discord](https://img.shields.io/discord/1377565354326495283?logo=discord&label=discord&labelColor=fff&color=5865f2)](https://discord.gg/tUmFEcYU4q)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mikel-brostrom/boxmot)

</div>

BoxMOT gives you one CLI and one Python API for running modern multi-object tracking workflows. It covers direct tracking, cached benchmark evaluation, tuning, research loops, and ReID export without forcing you to rebuild the detector and tracker stack for each experiment.

<div align="center" markdown="1">

[Docs](docs/index.md) • [Installation](docs/getting-started/installation.md) • [Modes](docs/modes/index.md) • [API Reference](docs/python/index.md) • [Trackers](docs/trackers/index.md) • [Contributing](CONTRIBUTING.md)

</div>

## Why BoxMOT

- One interface for `track`, `generate`, `eval`, `tune`, `research`, and `export`.
- Swappable trackers with shared detector and ReID plumbing.
- Benchmark-oriented workflows with reusable detections and embeddings.
- Support for both AABB and OBB tracking paths.
- Public Python API for embedding the same workflows in applications and notebooks.

## Installation

BoxMOT supports Python `3.9` through `3.12`.

```bash
pip install boxmot
boxmot --help
```

For mode-specific extras such as `yolo`, `evolve`, `research`, `onnx`, `openvino`, and `tflite`, see the [installation guide](docs/getting-started/installation.md).

## Benchmark Results (MOT17 ablation split)

<div align="center" markdown="1">

<!-- START TRACKER TABLE -->
| Tracker | Status  | OBB | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :-: | :---: | :---: | :---: | :---: |
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | ✅ | 69.418 | 78.232 | 81.812 | 12 |
| [boosttrack](https://arxiv.org/abs/2408.13003) | ✅ | ❌ | 69.253 | 75.914 | 83.206 | 13 |
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | ❌ | 68.05 | 76.185 | 80.763 | 11 |
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ | ❌ | 67.796 | 75.868 | 80.514 | 12 |
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | ✅ | 67.68 | 78.039 | 79.157 | 720 |
| [hybridsort](https://arxiv.org/abs/2308.00783) | ✅ | ❌ | 67.39 | 74.127 | 79.105 | 25 |
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ | ✅ | 66.441 | 74.548 | 77.899 | 890 |
| [sfsort](https://arxiv.org/pdf/2404.07553) | ✅ | ✅ | 62.653 | 76.87 | 69.184 | 6000 |
<!-- END TRACKER TABLE -->

<sub>Evaluation was run on the second half of the MOT17 training set because the validation split is not public and the ablation detector was trained on the first half. Results used [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip) with each tracker configured from its default repository settings.</sub>

</div>

Reproduction details and evaluation semantics live in:

- [Evaluation and Postprocessing](docs/guides/evaluation.md)
- [Benchmark Workflows](docs/guides/benchmarks.md)

## Minimal Usage

CLI:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
```

Python:

```python
from boxmot.api import Boxmot

run = Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort").track(
    source="video.mp4",
    save=True,
)
print(run)
```

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the [contributor docs](docs/contributing/index.md).

## Contributors

<a href="https://github.com/mikel-brostrom/boxmot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mikel-brostrom/boxmot" alt="BoxMOT contributors">
</a>

## Support and Citation

- Bugs and feature requests: [GitHub Issues](https://github.com/mikel-brostrom/boxmot/issues)
- Questions and discussion: [GitHub Discussions](https://github.com/mikel-brostrom/boxmot/discussions) or [Discord](https://discord.gg/tUmFEcYU4q)
- Citation metadata: [CITATION.cff](https://github.com/mikel-brostrom/boxmot/blob/master/CITATION.cff)
- Commercial support: `box-mot@outlook.com`
