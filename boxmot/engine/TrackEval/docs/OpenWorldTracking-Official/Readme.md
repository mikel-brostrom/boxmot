![owt](https://user-images.githubusercontent.com/23000532/160293694-6fc0a3da-c177-4776-8472-49ff6ff375a3.jpg)
# Opening Up Open-World Tracking - Official Evaluation Code

TrackEval now contains the official evalution code for evaluating the task of **Open World Tracking**.

This is the official code from the following paper:

<pre><b>Opening up Open-World Tracking</b>
Yang Liu*, Idil Esen Zulfikar*, Jonathon Luiten*, Achal Dave*, Deva Ramanan, Bastian Leibe, Aljoša Ošep, Laura Leal-Taixé
<t><t>*Equal contribution
CVPR 2022</pre>

[Paper](https://arxiv.org/abs/2104.11221)

[Website](https://openworldtracking.github.io)

## Running and understanding the code

The code can be run by running the following script (see script for arguments and how to run):
[TAO-OW run script](https://github.com/JonathonLuiten/TrackEval/blob/master/scripts/run_tao_ow.py)

To understand the the data is being read and used, see the TAO-OW dataset class:
[TAO-OW dataset class](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/tao_ow.py)

The implementation of the 'Open World Tracking Accuracy' (OWTA) metric proposed in the paper can be found here:
[OWTA metric](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py)

## Citation
If you work with the code and the benchmark, please cite:

***Opening Up Open-World Tracking***
```
@inproceedings{liu2022opening,
  title={Opening up Open-World Tracking},
  author={Liu, Yang and Zulfikar, Idil Esen and Luiten, Jonathon and Dave, Achal and Ramanan, Deva and Leibe, Bastian and O{\v{s}}ep, Aljo{\v{s}}a and Leal-Taix{\'e}, Laura},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

***TrackEval***
```
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```
