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
- Optional production-ready native C++ tracker implementations with the same metrics as the Python path, opted into via `--tracker-backend cpp` and embeddable in standalone C++ projects via CMake (see [Native C++ Integration](docs/guides/native-cpp.md)).
- Public Python API for embedding the same workflows in applications and notebooks.

## Installation

BoxMOT supports Python `3.10` through `3.13`.

```bash
pip install boxmot
boxmot --help
```

For mode-specific extras such as `yolo`, `evolve`, `research`, `onnx`, `openvino`, and `tflite`, see the [installation guide](docs/getting-started/installation.md).

## Benchmark Results

<div align="center" markdown="1">

<!-- START TRACKER TABLE -->
<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Tracker</th>
      <th rowspan="2" align="center">|</th>
      <th colspan="3" align="center">MOT17 (ablation split)</th>
      <th rowspan="2" align="center">|</th>
      <th colspan="3" align="center">SportsMOT (val split)</th>
      <th rowspan="2" align="center">|</th>
      <th colspan="3" align="center">MMOT (test split)</th>
      <th rowspan="2" align="center">|</th>
      <th rowspan="2" align="center">OBB</th>
    </tr>
    <tr>
      <th align="right">HOTA</th>
      <th align="right">MOTA</th>
      <th align="right">IDF1</th>
      <th align="right">HOTA</th>
      <th align="right">MOTA</th>
      <th align="right">IDF1</th>
      <th align="right">HOTA</th>
      <th align="right">MOTA</th>
      <th align="right">IDF1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">occluboost</td>
      <td align="center">|</td>
      <td align="right"><b>70.47</b></td>
      <td align="right"><b>78.32</b></td>
      <td align="right"><b>84.14</b></td>
      <td align="center">|</td>
      <td align="right"><b>83.17</b></td>
      <td align="right">97.48</td>
      <td align="right"><b>89.36</b></td>
      <td align="center">|</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="center">|</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/abs/2206.14651">botsort</a></td>
      <td align="center">|</td>
      <td align="right">69.44</td>
      <td align="right">78.24</td>
      <td align="right">81.94</td>
      <td align="center">|</td>
      <td align="right">76.93</td>
      <td align="right"><b>98.11</b></td>
      <td align="right">78.30</td>
      <td align="center">|</td>
      <td align="right"><b>51.79</b></td>
      <td align="right"><b>46.05</b></td>
      <td align="right"><b>60.85</b></td>
      <td align="center">|</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/abs/2408.13003">boosttrack</a></td>
      <td align="center">|</td>
      <td align="right">69.25</td>
      <td align="right">75.91</td>
      <td align="right">83.20</td>
      <td align="center">|</td>
      <td align="right">76.32</td>
      <td align="right">97.08</td>
      <td align="right">77.82</td>
      <td align="center">|</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="center">|</td>
      <td align="center">❌</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/abs/2202.13514">strongsort</a></td>
      <td align="center">|</td>
      <td align="right">68.05</td>
      <td align="right">76.19</td>
      <td align="right">80.76</td>
      <td align="center">|</td>
      <td align="right">79.80</td>
      <td align="right">97.31</td>
      <td align="right">80.27</td>
      <td align="center">|</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="center">|</td>
      <td align="center">❌</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/abs/2302.11813">deepocsort</a></td>
      <td align="center">|</td>
      <td align="right">67.95</td>
      <td align="right">75.83</td>
      <td align="right">80.54</td>
      <td align="center">|</td>
      <td align="right">79.51</td>
      <td align="right">97.94</td>
      <td align="right">79.59</td>
      <td align="center">|</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="center">|</td>
      <td align="center">❌</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/abs/2110.06864">bytetrack</a></td>
      <td align="center">|</td>
      <td align="right">67.68</td>
      <td align="right">78.04</td>
      <td align="right">79.16</td>
      <td align="center">|</td>
      <td align="right">67.93</td>
      <td align="right">97.25</td>
      <td align="right">76.90</td>
      <td align="center">|</td>
      <td align="right">33.97</td>
      <td align="right">33.72</td>
      <td align="right">39.74</td>
      <td align="center">|</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/abs/2308.00783">hybridsort</a></td>
      <td align="center">|</td>
      <td align="right">67.31</td>
      <td align="right">74.09</td>
      <td align="right">78.87</td>
      <td align="center">|</td>
      <td align="right">81.14</td>
      <td align="right">98.07</td>
      <td align="right">81.88</td>
      <td align="center">|</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="right">–</td>
      <td align="center">|</td>
      <td align="center">❌</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/abs/2203.14360">ocsort</a></td>
      <td align="center">|</td>
      <td align="right">66.44</td>
      <td align="right">74.55</td>
      <td align="right">77.90</td>
      <td align="center">|</td>
      <td align="right">76.34</td>
      <td align="right">96.60</td>
      <td align="right">75.64</td>
      <td align="center">|</td>
      <td align="right">28.57</td>
      <td align="right">26.19</td>
      <td align="right">29.95</td>
      <td align="center">|</td>
      <td align="center">✅</td>
    </tr>
    <tr>
      <td align="left"><a href="https://arxiv.org/pdf/2404.07553">sfsort</a></td>
      <td align="center">|</td>
      <td align="right">62.65</td>
      <td align="right">76.87</td>
      <td align="right">69.18</td>
      <td align="center">|</td>
      <td align="right">75.73</td>
      <td align="right">98.39</td>
      <td align="right">72.99</td>
      <td align="center">|</td>
      <td align="right">44.19</td>
      <td align="right">44.27</td>
      <td align="right">46.25</td>
      <td align="center">|</td>
      <td align="center">✅</td>
    </tr>
  </tbody>
</table>
<!-- END TRACKER TABLE -->

<sub>MOT17 evaluation was run on the second half of the MOT17 training set because the validation split is not public and the ablation detector was trained on the first half. Results used [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip) with each tracker configured from its default repository settings. SportsMOT evaluation used the val split with yolox_x_sportsmot detector and lmbn_n_duke ReID model. MMOT metrics are class-averaged across all 8 categories (car, bike, pedestrian, van, truck, bus, tricycle, awning-bike) following the <a href="https://arxiv.org/abs/2510.12565">MMOT paper</a> convention, using yolo11l_3ch detector and lmbn_n_duke ReID model.</sub>

</div>

Related guides:

- [Evaluation and Postprocessing](docs/guides/evaluation.md)
- [Benchmark Workflows](docs/guides/benchmarks.md)
- [Native C++ Integration](docs/guides/native-cpp.md)

## Minimal Usage

CLI:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
```

Python:

```python
from boxmot import Boxmot

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
