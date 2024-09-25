# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from typing import List, Tuple
from boxmot.utils.matching import iou_distance


def joint_stracks(tlista: List['STrack'], tlistb: List['STrack']) -> List['STrack']:
    """
    Joins two lists of tracks, ensuring that there are no duplicates based on track IDs.

    Args:
        tlista (List[STrack]): The first list of tracks.
        tlistb (List[STrack]): The second list of tracks.

    Returns:
        List[STrack]: A combined list of tracks from both input lists, without duplicates.
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: List['STrack'], tlistb: List['STrack']) -> List['STrack']:
    """
    Subtracts the tracks in tlistb from tlista based on track IDs.

    Args:
        tlista (List[STrack]): The list of tracks from which tracks will be removed.
        tlistb (List[STrack]): The list of tracks to be removed from tlista.

    Returns:
        List[STTrack]: The remaining tracks after removal.
    """
    stracks = {t.id: t for t in tlista}
    for t in tlistb:
        tid = t.id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa: List['STrack'], stracksb: List['STrack']) -> Tuple[List['STrack'], List['STrack']]:
    """
    Removes duplicate tracks between two lists based on their IoU distance and track duration.

    Args:
        stracksa (List[STrack]): The first list of tracks.
        stracksb (List[STrack]): The second list of tracks.

    Returns:
        Tuple[List[STrack], List[STrack]]: The filtered track lists, with duplicates removed.
    """
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    
    return resa, resb
