<div align="center" markdown="1">

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/logo_white.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo/logo_black.png">
    <img width="400"
         src="docs/logo/logo_black.png"
         alt="BoxMOT logo">
  </picture>

  <p><b>Pluggable Python and C++ multi-object tracking modules for axis-aligned and oriented bounding box detections from any model.</b></p>

  [![CI](https://github.com/mikel-brostrom/boxmot/actions/workflows/ci.yml/badge.svg)](https://github.com/mikel-brostrom/boxmot/actions/workflows/ci.yml)
  [![PyPI version](https://badge.fury.io/py/boxmot.svg)](https://badge.fury.io/py/boxmot)
  [![downloads](https://static.pepy.tech/badge/boxmot)](https://pepy.tech/project/boxmot)
  [![license](https://img.shields.io/badge/license-AGPL%203.0-blue)](https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE)
  [![python-version](https://img.shields.io/pypi/pyversions/boxmot)](https://badge.fury.io/py/boxmot)
  [![docker pulls](https://img.shields.io/docker/pulls/boxmot/boxmot?logo=docker)](https://hub.docker.com/r/boxmot/boxmot)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8132989.svg)](https://doi.org/10.5281/zenodo.8132989)
  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing)
  [![discord](https://img.shields.io/discord/1377565354326495283?logo=discord&label=discord&labelColor=fff&color=5865f2)](https://discord.gg/tUmFEcYU4q)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mikel-brostrom/boxmot)

  <a href="https://trendshift.io/repositories/13239" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13239" alt="mikel-brostrom%2Fboxmot | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"></a>

  ---

  [Docs](docs/index.md) • [Installation](docs/getting-started/installation.md) • [Modes](docs/modes/index.md) • [API Reference](docs/python/index.md) • [Trackers](docs/trackers/index.md) • [Contributing](CONTRIBUTING.md)

  <img width="640"
       src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif"
       alt="BoxMOT demo">

</div>

BoxMOT gives you one CLI and one Python API for running modern multi-object tracking workflows. It covers direct tracking, catalog-backed evaluation, tuning, research loops, ReID training and evaluation, and ReID export without forcing you to rebuild the detector and tracker stack for each experiment.

## Why BoxMOT

- One interface for `track`, `generate`, `eval`, `tune`, `research`,
  `train-reid`, `eval-reid`, `export`, and native `build`
  workflows.
- Swappable trackers with shared detector and ReID plumbing.
- Dataset and experiment workflows with reusable detections and embeddings.
- Support for both AABB and OBB tracking paths.
- Optional production-ready native C++ tracker implementations with the same metrics as the Python path, opted into via `--tracker-backend cpp` and embeddable in standalone C++ projects via CMake (see [Native C++ Integration](docs/native/index.md)).
- Public Python API for embedding the same workflows in applications and notebooks.

## Installation

BoxMOT supports Python `3.10` through `3.13`.

```bash
pip install boxmot
boxmot --help
```

The default package uses the standard PyPI PyTorch build. Source checkouts and
CI can explicitly select the lockfile-backed `cpu` or `cu130` profile. For
those profiles and mode-specific extras such as `yolo`, `service`, `evolve`,
`research`, `onnx`, `openvino`, and `tflite`, see the
[installation guide](docs/getting-started/installation.md).

## Docker images

Published images cover GPU and CPU CLI workflows plus separate CPU geometry
and GPU ReID tracker services:

```bash
# GPU-enabled detector, CLI, evaluation, and interactive workflows
docker run --rm -it --gpus all boxmot/boxmot:latest

# The same CLI workflows on CPU
docker run --rm -it boxmot/boxmot:latest-cpu

# CPU-only stateful HTTP tracking from externally supplied detections
docker run --rm -p 8000:8000 \
  -e BOXMOT_SERVICE_ASSO_FUNC=giou \
  boxmot/boxmot-service:latest

# CUDA/ReID stateful tracking from detections plus an encoded image per frame
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:latest-gpu
```

Versioned and commit-addressed tags are also published. GPU CLI tags are
`<version>` and `sha-<commit>`; CPU CLI tags append `-cpu`. The CPU service uses
canonical `<version>` and `sha-<commit>` tags in its own repository, while the
GPU service appends `-gpu`. See the
[installation guide](docs/getting-started/installation.md) for local builds.

Both services accept ordered AABB or OBB detections and keep isolated state per
stream/session; neither runs a detector. The CPU image supports ByteTrack,
OCSort, and SFSORT without image pixels. The GPU image supports StrongSORT,
BotSORT, DeepOCSORT, HybridSORT, BoostTrack, and OccluBoost, and requires a raw
base64-encoded JPEG or PNG in `image_base64` for every frame, including empty
detection frames. See the [deployment guide](docs/guides/deployment.md) for the
request schema and horizontal-scaling requirements.

## Benchmark Results

<div align="center" markdown="1">

<!-- START TRACKER TABLE -->
<table>
  <thead>
    <tr>
      <th rowspan="2" align="left"><sub>Tracker</sub></th>
      <th rowspan="2" align="center"><sub>Status</sub></th>
      <th colspan="3" align="center"><sub>MOT17 ablation</sub></th>
      <th colspan="3" align="center"><sub>SportsMOT val</sub></th>
      <th colspan="3" align="center"><sub>MMOT OBB test</sub></th>
      <th rowspan="2" align="center"><sub>OBB</sub></th>
    </tr>
    <tr>
      <th align="right"><sub>HOTA</sub></th>
      <th align="right"><sub>MOTA</sub></th>
      <th align="right"><sub>IDF1</sub></th>
      <th align="right"><sub>HOTA</sub></th>
      <th align="right"><sub>MOTA</sub></th>
      <th align="right"><sub>IDF1</sub></th>
      <th align="right"><sub>HOTA</sub></th>
      <th align="right"><sub>MOTA</sub></th>
      <th align="right"><sub>IDF1</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><sub>occluboost</sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub><b>71.10</b><br>(71.10)</sub></td>
      <td align="right"><sub><b>78.50</b><br>(78.50)</sub></td>
      <td align="right"><sub><b>85.28</b><br>(85.28)</sub></td>
      <td align="right"><sub><b>83.17</b></sub></td>
      <td align="right"><sub>97.48</sub></td>
      <td align="right"><sub><b>89.36</b></sub></td>
      <td align="right"><sub>49.84<br>(49.84)</sub></td>
      <td align="right"><sub>39.41<br>(39.41)</sub></td>
      <td align="right"><sub>58.60<br>(58.60)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/abs/2206.14651">botsort</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>69.68<br>(69.74)</sub></td>
      <td align="right"><sub>78.23<br>(78.27)</sub></td>
      <td align="right"><sub>82.33<br>(82.55)</sub></td>
      <td align="right"><sub>76.93</sub></td>
      <td align="right"><sub><b>98.11</b></sub></td>
      <td align="right"><sub>78.30</sub></td>
      <td align="right"><sub>52.31<br>(52.40)</sub></td>
      <td align="right"><sub>45.43<br>(45.53)</sub></td>
      <td align="right"><sub>61.42<br>(61.42)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/abs/2408.13003">boosttrack</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>69.25<br>(—)</sub></td>
      <td align="right"><sub>75.91<br>(—)</sub></td>
      <td align="right"><sub>83.20<br>(—)</sub></td>
      <td align="right"><sub>76.32</sub></td>
      <td align="right"><sub>97.08</sub></td>
      <td align="right"><sub>77.82</sub></td>
      <td align="right"><sub>48.39<br>(—)</sub></td>
      <td align="right"><sub>41.36<br>(—)</sub></td>
      <td align="right"><sub>56.36<br>(—)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/abs/2202.13514">strongsort</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>68.05<br>(—)</sub></td>
      <td align="right"><sub>76.19<br>(—)</sub></td>
      <td align="right"><sub>80.76<br>(—)</sub></td>
      <td align="right"><sub>79.80</sub></td>
      <td align="right"><sub>97.31</sub></td>
      <td align="right"><sub>80.27</sub></td>
      <td align="right"><sub>49.76<br>(—)</sub></td>
      <td align="right"><sub>43.70<br>(—)</sub></td>
      <td align="right"><sub>57.32<br>(—)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/abs/2302.11813">deepocsort</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>67.95<br>(—)</sub></td>
      <td align="right"><sub>75.83<br>(—)</sub></td>
      <td align="right"><sub>80.54<br>(—)</sub></td>
      <td align="right"><sub>79.51</sub></td>
      <td align="right"><sub>97.94</sub></td>
      <td align="right"><sub>79.59</sub></td>
      <td align="right"><sub>50.84<br>(—)</sub></td>
      <td align="right"><sub>44.21<br>(—)</sub></td>
      <td align="right"><sub>59.33<br>(—)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/abs/2110.06864">bytetrack</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>67.68<br>(67.68)</sub></td>
      <td align="right"><sub>78.04<br>(78.04)</sub></td>
      <td align="right"><sub>79.16<br>(79.16)</sub></td>
      <td align="right"><sub>67.93</sub></td>
      <td align="right"><sub>97.25</sub></td>
      <td align="right"><sub>76.90</sub></td>
      <td align="right"><sub>33.97<br>(33.97)</sub></td>
      <td align="right"><sub>33.72<br>(33.72)</sub></td>
      <td align="right"><sub>39.74<br>(39.74)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/abs/2308.00783">hybridsort</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>67.31<br>(—)</sub></td>
      <td align="right"><sub>74.09<br>(—)</sub></td>
      <td align="right"><sub>78.87<br>(—)</sub></td>
      <td align="right"><sub>81.14</sub></td>
      <td align="right"><sub>98.07</sub></td>
      <td align="right"><sub>81.88</sub></td>
      <td align="right"><sub><b>54.64</b><br>(—)</sub></td>
      <td align="right"><sub><b>47.50</b><br>(—)</sub></td>
      <td align="right"><sub><b>64.67</b><br>(—)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/abs/2203.14360">ocsort</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>66.44<br>(66.44)</sub></td>
      <td align="right"><sub>74.55<br>(74.55)</sub></td>
      <td align="right"><sub>77.90<br>(77.90)</sub></td>
      <td align="right"><sub>76.34</sub></td>
      <td align="right"><sub>96.60</sub></td>
      <td align="right"><sub>75.64</sub></td>
      <td align="right"><sub>28.64<br>(28.64)</sub></td>
      <td align="right"><sub>26.17<br>(26.17)</sub></td>
      <td align="right"><sub>30.06<br>(30.06)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
    <tr>
      <td align="left"><sub><a href="https://arxiv.org/pdf/2404.07553">sfsort</a></sub></td>
      <td align="center"><sub>✅</sub></td>
      <td align="right"><sub>62.65<br>(62.65)</sub></td>
      <td align="right"><sub>76.87<br>(76.87)</sub></td>
      <td align="right"><sub>69.18<br>(69.18)</sub></td>
      <td align="right"><sub>75.73</sub></td>
      <td align="right"><sub>98.39</sub></td>
      <td align="right"><sub>72.99</sub></td>
      <td align="right"><sub>47.83<br>(47.83)</sub></td>
      <td align="right"><sub>45.42<br>(45.42)</sub></td>
      <td align="right"><sub>52.09<br>(52.09)</sub></td>
      <td align="center"><sub>✅</sub></td>
    </tr>
  </tbody>
</table>
<!-- END TRACKER TABLE -->

<p align="center">
  <sub>Scores are Python first and C++ in parentheses.</sub><br>
  <sub>MMOT reported metrics are 'class average'. See <a href="docs/guides/experiments.md">Experiment Workflows</a> for details.</sub>
</p>

</div>

Related guides:

- [Evaluation and Postprocessing](docs/guides/evaluation.md)
- [Experiment Workflows](docs/guides/experiments.md)
- [Native C++ Integration](docs/native/index.md)

## Minimal Usage

CLI:

```bash
boxmot track --detector yolo26n --reid lmbn_n_duke --tracker occluboost \
  --asso-func diou --source 0 --save --show
```

Python:

```python
import numpy as np
from boxmot.trackers.registry import create_tracker

tracker = create_tracker(
    "occluboost",
    reid_weights="osnet_x0_25_msmt17.pt",
    device="cpu",
    half=False,
    tracker_kwargs={"asso_func": "diou"},
)

# dets: (N, 6) array with [x1, y1, x2, y2, conf, cls] per detection
dets = np.array([[100, 200, 300, 400, 0.9, 0]], dtype=np.float32)
# OBB alternative: (N, 7) with [cx, cy, w, h, angle_radians, conf, cls]
# dets = np.array([[200, 300, 200, 200, 0.25, 0.9, 0]], dtype=np.float32)
img = np.zeros((480, 640, 3), dtype=np.uint8)  # current frame

# tracks: AABB (M, 8), or OBB (M, 9) with angle after h
tracks = tracker.update(dets, img=img)
print(tracks)
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
- Limited free consulting is available for nonprofit nature conservation projects using BoxMOT. Contact `box-mot@outlook.com` to discuss your project.
- Citation metadata: [CITATION.cff](https://github.com/mikel-brostrom/boxmot/blob/master/CITATION.cff)
- Commercial support: `box-mot@outlook.com`
