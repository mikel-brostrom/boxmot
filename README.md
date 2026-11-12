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
  [![Coverage](https://codecov.io/gh/mikel-brostrom/boxmot/branch/master/graph/badge.svg)](https://app.codecov.io/gh/mikel-brostrom/boxmot/tree/master)
  [![PyPI version](https://badge.fury.io/py/boxmot.svg)](https://badge.fury.io/py/boxmot)
  [![downloads](https://static.pepy.tech/badge/boxmot)](https://pepy.tech/project/boxmot)
  [![license](https://img.shields.io/badge/license-AGPL%203.0-blue)](https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE)
  [![python-version](https://img.shields.io/pypi/pyversions/boxmot)](https://badge.fury.io/py/boxmot)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8132989.svg)](https://doi.org/10.5281/zenodo.8132989)
  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing)
  [![discord](https://img.shields.io/discord/1377565354326495283?logo=discord&label=discord&labelColor=fff&color=5865f2)](https://discord.gg/tUmFEcYU4q)
  [<img src="https://devin.ai/assets/askdeepwiki.png" alt="Ask DeepWiki" height="20">](https://deepwiki.com/mikel-brostrom/boxmot)

  <a href="https://trendshift.io/repositories/13239" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13239" alt="mikel-brostrom%2Fboxmot | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"></a>

  ---

  [Docs](docs/index.md) • [Installation](docs/getting-started/installation.md) • [Modes](docs/modes/index.md) • [API Reference](docs/python/index.md) • [Trackers](docs/trackers/index.md) • [Datasets](docs/datasets/index.md) • [Contributing](CONTRIBUTING.md)

  <img width="640"
       src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif"
       alt="BoxMOT demo">

</div>

BoxMOT provides independent detector, segmentor, appearance-encoder, and tracker
components built around validated Torch structures. Pipelines compose those
components; the CLI owns sources, outputs, materialized datasets, evaluation,
tuning, research, and ReID workflows.

## Why BoxMOT

- One interface for `track`, `materialize`, `materialize --time-variant`, `eval`,
  `tune`, `research`, `train-reid`, `eval-reid`, `compare-reid`, `export`,
  `install`, and native `build` workflows.
- Swappable components with explicit capabilities and requirements.
- Immutable, keyed Parquet builds with reusable detections, masks, and
  embeddings.
- Support for both AABB and OBB tracking paths.
- Optional native C++ tracker implementations selected with `--tracker-backend cpp`, with shared detection/output contracts and documented backend differences. They are also embeddable in standalone C++ projects through CMake (see [Native C++ Integration](docs/native/index.md) and [platform dependency installation](docs/native/index.md#install-dependencies)).
- A structured Python API for embedding components and pipelines in applications.

## Installation

Requires Python `3.10`–`3.13`.

Install from PyPI:

```bash
pip install boxmot "ultralytics>=8.4.58"
boxmot --help
```

Or, from a source checkout, run in the repository root:

```bash
pip install "uv==0.12.4"
uv sync --extra cpu --extra yolo
uv run --no-sync python -m boxmot.engine.cli --help
```

Replace `--extra cpu` with `--extra cu130` for CUDA 13.0.

See [installation options](docs/getting-started/installation.md),
[source and CPU/CUDA setup](docs/getting-started/pytorch.md), and
[GPU batch sizing](docs/modes/materialize.md#automatic-gpu-batch-sizing).

## Benchmark Results

<div align="center" markdown="1">

<!-- START TRACKER TABLE -->
<table>
  <thead>
    <tr>
      <th rowspan="2" align="left"><sub>Tracker key</sub></th>
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

Choose a workflow and settings with the [terminal UI](docs/modes/ui.md):

```bash
boxmot ui
```

Track a webcam from the CLI:

```bash
boxmot track --detector yolo26n --reid lmbn_n_duke --tracker occluboost \
  --source 0 --save --show
```

Python with axis-aligned boxes (AABB):

```python
import numpy as np

from boxmot import OccluBoost

tracker = OccluBoost()
# x1, y1, x2, y2, confidence, class_id
dets = np.array([[100, 200, 300, 400, 0.9, 0]])
frame = np.zeros((480, 640, 3), dtype=np.uint8)  # BGR image
tracks = tracker.update(dets, frame)
print(tracks[:, 4].astype(int))  # track IDs
```

Python with [oriented boxes (OBB)](docs/tasks/obb.md):

```python
import numpy as np

from boxmot import OccluBoost

tracker = OccluBoost(is_obb=True)
# cx, cy, width, height, angle in radians, confidence, class_id
dets = np.array([[200, 300, 200, 100, np.pi / 6, 0.9, 0]])
frame = np.zeros((480, 640, 3), dtype=np.uint8)  # BGR image
tracks = tracker.update(dets, frame)
print(tracks[:, 5].astype(int))  # track IDs
```

See the [mode guides](docs/modes/index.md) for more CLI and Python workflows.
Advanced examples: [capture timing](docs/python/index.md#elapsed-time),
[prompted tracking with EdgeTAM](docs/trackers/edgetam.md), and
[2D/3D fusion with EagerMot](docs/trackers/eagermot.md).

## Track Attributes with Ollama and vLLM

<table>
  <thead>
    <tr>
      <th>Provider</th>
      <th>Description</th>
      <th>Pros</th>
      <th>Cons</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://ollama.com">
          <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ollama/ollama/main/docs/images/logo-dark.png">
            <img src="https://raw.githubusercontent.com/ollama/ollama/main/docs/images/logo.png" alt="Ollama" height="48">
          </picture>
        </a>
      </td>
      <td>Easily run vision models locally, with straightforward installation, model downloads, and API access.</td>
      <td>Simple setup; images stay on your machine when using local models.</td>
      <td>Large models need substantial RAM/VRAM and disk space; speed depends on your hardware.</td>
    </tr>
    <tr>
      <td align="center">
        <a href="https://vllm.ai">
          <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/media-kit/main/vLLM-Full-Dark-Mode-Logo.svg">
            <img src="https://raw.githubusercontent.com/vllm-project/media-kit/main/vLLM-Full-Logo.svg" alt="vLLM" height="48">
          </picture>
        </a>
      </td>
      <td>Serve vision-language models on your own infrastructure, with more configuration options for handling many requests efficiently.</td>
      <td>Efficient batching for concurrent requests; control over models and infrastructure.</td>
      <td>More setup, tuning, and maintenance; you provide the hardware to run the models.</td>
    </tr>
  </tbody>
</table>

Add clothing, footwear, and pose descriptions to person tracks with
**[Ollama](https://ollama.com)** or **[vLLM](https://vllm.ai)**.
Display attributes alongside track IDs
and export structured JSON results.
Every attribute run requires PostgreSQL for its queue and retained results;
the Compose workflow starts the database automatically.

See the [Track Attributes guide](docs/config/attributes.md) for setup,
configuration, and examples. To run detection, ReID, and the VLM on NVIDIA GPUs,
use the [GPU Compose setup](docs/integrations/docker.md#gpu-detection-reid-and-vlm-inference).

## License

BoxMOT is licensed under the [GNU Affero General Public License v3 (AGPL-3.0)](LICENSE).

Each detector and segmentor integrated with BoxMOT retains its own upstream
license. See the respective project's license for its terms. Other third-party
components also retain their respective licenses and notices.

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
