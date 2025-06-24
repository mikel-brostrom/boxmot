
# TrackEval
*Code for evaluating object tracking.*

This codebase provides code for a number of different tracking evaluation metrics (including the [HOTA metrics](https://link.springer.com/article/10.1007/s11263-020-01375-2)), as well as supporting running all of these metrics on a number of different tracking benchmarks. Plus plotting of results and other things one may want to do for tracking evaluation.

## **NEW**: RobMOTS Challenge 2021

Call for submission to our [RobMOTS Challenge](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110) (Robust Multi-Object Tracking and Segmentation) held in conjunction with our [RVSU CVPR'21 Workshop](https://eval.vision.rwth-aachen.de/rvsu-workshop21/). Robust tracking evaluation against 8 tracking benchmarks. Challenge submission deadline June 15th. Also check out our workshop [call for papers](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=74).

## Official Evaluation Code

The following benchmarks use TrackEval as their official evaluation code, check out the links to see TrackEval in action:

 - **[RobMOTS](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110)** ([Official Readme](docs/RobMOTS-Official/Readme.md))
 - **[KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)**
 - **[KITTI MOTS](http://www.cvlibs.net/datasets/kitti/eval_mots.php)**
 - **[MOTChallenge](https://motchallenge.net/)** ([Official Readme](docs/MOTChallenge-Official/Readme.md))
 - **[Open World Tracking](https://openworldtracking.github.io)** ([Official Readme](docs/OpenWorldTracking-Official))
 - **[PersonPath22](https://amazon-research.github.io/tracking-dataset/personpath22.html)**
 <!--- **[MOTS-Challenge](https://motchallenge.net/data/MOTS/)** ([Official Readme](docs/MOTS-Challenge-Official/Readme.md)) --->

If you run a tracking benchmark and want to use TrackEval as your official evaluation code, please contact Jonathon (contact details below).

## Currently implemented metrics

The following metrics are currently implemented:

Metric Family | Sub metrics | Paper | Code | Notes |
|----- | ----------- |----- | ----------- | ----- |
| | | |  |  |
|**HOTA metrics**|HOTA, DetA, AssA, LocA, DetPr, DetRe, AssPr, AssRe|[paper](https://link.springer.com/article/10.1007/s11263-020-01375-2)|[code](trackeval/metrics/hota.py)|**Recommended tracking metric**|
|**CLEARMOT metrics**|MOTA, MOTP, MT, ML, Frag, etc.|[paper](https://link.springer.com/article/10.1155/2008/246309)|[code](trackeval/metrics/clear.py)| |
|**Identity metrics**|IDF1, IDP, IDR|[paper](https://arxiv.org/abs/1609.01775)|[code](trackeval/metrics/identity.py)| |
|**VACE metrics**|ATA, SFDA|[paper](https://link.springer.com/chapter/10.1007/11612704_16)|[code](trackeval/metrics/vace.py)| |
|**Track mAP metrics**|Track mAP|[paper](https://arxiv.org/abs/1905.04804)|[code](trackeval/metrics/track_map.py)|Requires confidence scores|
|**J & F metrics**|J&F, J, F|[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)|[code](trackeval/metrics/j_and_f.py)|Only for Seg Masks|
|**ID Euclidean**|ID Euclidean|[paper](https://arxiv.org/pdf/2103.13516.pdf)|[code](trackeval/metrics/ideucl.py)| |


## Currently implemented benchmarks

The following benchmarks are currently implemented:

Benchmark | Sub-benchmarks | Type | Website | Code | Data Format |
|----- | ----------- |----- | ----------- | ----- | ----- |
| | | |  |  | |
|**RobMOTS**|Combination of 8 benchmarks|Seg Masks|[website](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110)|[code](trackeval/datasets/rob_mots.py)|[format](docs/RobMOTS-Official/Readme.md)|
|**Open World Tracking**|TAO-OW|OpenWorld / Seg Masks|[website](https://openworldtracking.github.io)|[code](trackeval/datasets/tao_ow.py)|[format](docs/OpenWorldTracking-Official/Readme.md)|
|**MOTChallenge**|MOT15/16/17/20|2D BBox|[website](https://motchallenge.net/)|[code](trackeval/datasets/mot_challenge_2d_box.py)|[format](docs/MOTChallenge-format.txt)|
|**KITTI Tracking**| |2D BBox|[website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)|[code](trackeval/datasets/kitti_2d_box.py)|[format](docs/KITTI-format.txt)|
|**BDD-100k**| |2D BBox|[website](https://bdd-data.berkeley.edu/)|[code](trackeval/datasets/bdd100k.py)|[format](docs/BDD100k-format.txt)|
|**TAO**| |2D BBox|[website](https://taodataset.org/)|[code](trackeval/datasets/tao.py)|[format](docs/TAO-format.txt)|
|**MOTS**|KITTI-MOTS, MOTS-Challenge|Seg Mask|[website](https://www.vision.rwth-aachen.de/page/mots)|[code](trackeval/datasets/mots_challenge.py) and [code](trackeval/datasets/kitti_mots.py)|[format](docs/MOTS-format.txt)|
|**DAVIS**|Unsupervised|Seg Mask|[website](https://davischallenge.org/)|[code](trackeval/datasets/davis.py)|[format](docs/DAVIS-format.txt)|
|**YouTube-VIS**| |Seg Mask|[website](https://youtube-vos.org/dataset/vis/)|[code](trackeval/datasets/youtube_vis.py)|[format](docs/YouTube-VIS-format.txt)|
|**Head Tracking Challenge**| |2D BBox|[website](https://arxiv.org/pdf/2103.13516.pdf)|[code](trackeval/datasets/head_tracking_challenge.py)|[format](docs/MOTChallenge-format.txt)|
|**PersonPath22**| |2D BBox|[website](https://github.com/amazon-research/tracking-dataset)|[code](trackeval/datasets/person_path_22.py)|[format](docs/MOTChallenge-format.txt)|
|**BURST**| {Common, Long-tail, Open-world} Class-guided, {Point, Box, Mask} Exemplar-guided |Seg Mask|[website](https://github.com/Ali2500/BURST-benchmark)|[format](https://github.com/Ali2500/BURST-benchmark/blob/main/ANNOTATION_FORMAT.md)|

## HOTA metrics

This code is also the official reference implementation for the HOTA metrics:

*[HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2). IJCV 2020. Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe and Bastian Leibe.*

HOTA is a novel set of MOT evaluation metrics which enable better understanding of tracking behavior than previous metrics.

For more information check out the following links:
 - [Short blog post on HOTA](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1) - **HIGHLY RECOMMENDED READING**
 - [IJCV version of paper](https://link.springer.com/article/10.1007/s11263-020-01375-2) (Open Access)
 - [ArXiv version of paper](https://arxiv.org/abs/2009.07736)
 - [Code](trackeval/metrics/hota.py)

## Properties of this codebase

The code is written 100% in python with only numpy and scipy as minimum requirements.

The code is designed to be easily understandable and easily extendable. 

The code is also extremely fast, running at more than 10x the speed of the both [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit), and [py-motmetrics](https://github.com/cheind/py-motmetrics) (see detailed speed comparison below).

The implementation of CLEARMOT and ID metrics aligns perfectly with the [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

By default the code prints results to the screen, saves results out as both a summary txt file and a detailed results csv file, and outputs plots of the results. All outputs are by default saved to the 'tracker' folder for each tracker.

## Running the code

The code can be run in one of two ways:

 - From the terminal via one of the scripts [here](scripts/). See each script for instructions and arguments, hopefully this is self-explanatory.
 - Directly by importing this package into your code, see the same scripts above for how. 

## Quickly evaluate on supported benchmarks

To enable you to use TrackEval for evaluation as quickly and easily as possible, we provide ground-truth data, meta-data and example trackers for all currently supported benchmarks.
You can download this here: [data.zip](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) (~150mb).

The data for RobMOTS is separate and can be found here: [rob_mots_train_data.zip](https://omnomnom.vision.rwth-aachen.de/data/RobMOTS/train_data.zip) (~750mb).

The data for PersonPath22 is separate and can be found here: [person_path_22_data.zip](https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/person_path_22_data.zip) (~3mb).

The easiest way to begin is to extract this zip into the repository root folder such that the file paths look like: TrackEval/data/gt/...

This then corresponds to the default paths in the code. You can now run each of the scripts [here](scripts/) without providing any arguments and they will by default evaluate all trackers present in the supplied file structure. To evaluate your own tracking results, simply copy your files as a new tracker folder into the file structure at the same level as the example trackers (MPNTrack, CIWT, track_rcnn, qdtrack, ags, Tracktor++, STEm_Seg), ensuring the same file structure for your trackers as in the example.

Of course, if your ground-truth and tracker files are located somewhere else you can simply use the script arguments to point the code toward your data.

To ensure your tracker outputs data in the correct format, check out our format guides for each of the supported benchmarks [here](docs), or check out the example trackers provided.

## Evaluate on your own custom benchmark

To evaluate on your own data, you have two options:
 - Write custom dataset code (more effort, rarely worth it).
 - Convert your current dataset and trackers to the same format of an already implemented benchmark.

To convert formats, check out the format specifications defined [here](docs).

By default, we would recommend the MOTChallenge format, although any implemented format should work. Note that for many cases you will want to use the argument ```--DO_PREPROC False``` unless you want to run preprocessing to remove distractor objects.

## Requirements
 Code tested on Python 3.7.
 
 - Minimum requirements: numpy, scipy
 - For plotting: matplotlib
 - For segmentation datasets (KITTI MOTS, MOTS-Challenge, DAVIS, YouTube-VIS): pycocotools
 - For DAVIS dataset: Pillow
 - For J & F metric: opencv_python, scikit_image
 - For simples test-cases for metrics: pytest

use ```pip3 -r install requirements.txt``` to install all possible requirements.

use ```pip3 -r install minimum_requirments.txt``` to only install the minimum if you don't need the extra functionality as listed above.

## Timing analysis

Evaluating CLEAR + ID metrics on Lift_T tracker on MOT17-train (seconds) on a i7-9700K CPU with 8 physical cores (median of 3 runs):		
Num Cores|TrackEval|MOTChallenge|Speedup vs MOTChallenge|py-motmetrics|Speedup vs py-motmetrics
:---|:---|:---|:---|:---|:---
1|9.64|66.23|6.87x|99.65|10.34x
4|3.01|29.42|9.77x| |33.11x*
8|1.62|29.51|18.22x| |61.51x*

*using a different number of cores as py-motmetrics doesn't allow multiprocessing.
				
```
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --TRACKERS_TO_EVAL Lif_T --METRICS CLEAR Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1  
```
				
Evaluating CLEAR + ID metrics on LPC_MOT tracker on MOT20-train (seconds) on a i7-9700K CPU with 8 physical cores (median of 3 runs):	
Num Cores|TrackEval|MOTChallenge|Speedup vs MOTChallenge|py-motmetrics|Speedup vs py-motmetrics
:---|:---|:---|:---|:---|:---
1|18.63|105.3|5.65x|175.17|9.40x

```
python scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL LPC_MOT --METRICS CLEAR Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```

## License

TrackEval is released under the [MIT License](LICENSE).

## Contact

If you encounter any problems with the code, please contact [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/) ([luiten@vision.rwth-aachen.de](mailto:luiten@vision.rwth-aachen.de)).
If anything is unclear, or hard to use, please leave a comment either via email or as an issue and I would love to help.

## Dedication

This codebase was built for you, in order to make your life easier! For anyone doing research on tracking or using trackers, please don't hesitate to reach out with any comments or suggestions on how things could be improved.

## Contributing

We welcome contributions of new metrics and new supported benchmarks. Also any other new features or code improvements. Send a PR, an email, or open an issue detailing what you'd like to add/change to begin a conversation.

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
