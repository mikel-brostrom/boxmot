from __future__ import division, print_function, absolute_import


from timeit import time
import warnings

import numpy as np
import base64
import requests
import random
import time
import operator

def return_people_v2(reid, feats_all):
    final_fuse_id = {0 : [], 1 : []}
    threshold = 320
    index = 0
    people_count = 0
    for feature in feats_all:
        for f in feature:
            print(final_fuse_id)
            if index == 0 and len(final_fuse_id[index]) == 0:
                final_fuse_id[index].append(f)
                people_count += 1
            else :
                dis = []
                for key, item in final_fuse_id.items():
                    for i in item:
                        tmp = np.mean(reid.compute_distance(feats_all[index][f], feats_all[key][i]))
                        dis.append(tmp)
                dis.sort()
                if dis[0] >= threshold:
                    people_count += 1
                    final_fuse_id[index].append(f)
        index += 1
    return people_count
