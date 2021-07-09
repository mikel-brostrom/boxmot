from __future__ import division, print_function, absolute_import


from timeit import time
import warnings

import numpy as np
import base64
import requests
import random
import time
import operator

from reid import REID


def return_people(reid, images_by_id, ids_per_frame):
    exist_ids = set()
    final_fuse_id = dict()
    threshold = 320
    time_re = time.time()
    feats = dict()
    for i in images_by_id:
        feats[i] = reid._features(images_by_id[i])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])

    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids,unpickable))
                    for oid in (exist_ids-set(unpickable))&set(final_fuse_id.keys()):
                        tmp = np.mean(reid.compute_distance(feats[nid],feats[oid]))
                        print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]
    print('People counting : {}, ReID tracking : {}'.format(len(final_fuse_id), time.time() - time_re))
    print('People id: {}'.format(final_fuse_id))
    return len(final_fuse_id)
