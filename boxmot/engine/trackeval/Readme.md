# TrackEval for MMOT

This document explains how to evaluate tracking results on **MMOT** using the
wrapper script `eval.sh` (which calls `scripts/run_mmot_8ch.py` under the hood).

The toolkit computes **HOTA**, **MOTA**, **IDF1**, and **CLEAR** metrics and supports
both **oriented** MOT results produced by all modules in this repo
(MOTR / MOTRv2 / MeMOTR / MOTIP / SORT / ByteTrack / OC-SORT / BoT-SORT). Some methods have already included evaluation steps in their own scripts.


## ▶️ Quick Start


`eval.sh` has the following calling convention:

```bash
cd TrackEval
bash eval.sh <TRACKERS_TO_EVAL> <TRACKERS_SUB_FOLDER> [TRACKERS_FOLDER]

```
> If your results are saved elsewhere, pass the absolute path as the **third**
argument of `eval.sh`.

### Examples
```bash
# Make attention that different trackers may save results in different subfolder.
bash eval.sh motr/train_8ch-3D preds
bash eval.sh motrv2/train_8ch-3D submit
bash eval.sh memotr/8ch3d test/tracker 

```


## Citing TrackEval

If you use this code in your research, please use the following BibTeX entry:

```BibTeX
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```

Furthermore, if you use the HOTA metrics, please cite the following paper:

```
@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```

If you use any other metrics please also cite the relevant papers, and don't forget to cite each of the benchmarks you evaluate on.
