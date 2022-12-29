from trackers.strongsort.utils.parser import get_config
from trackers.strongsort.strong_sort import StrongSORT
from trackers.ocsort.ocsort import OCSort
from trackers.bytetrack.byte_tracker import BYTETracker


def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    
    cfg = get_config()
    cfg.merge_from_file(tracker_config)
    
    if tracker_type == 'strongsort':

        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            max_unmatched_preds=cfg.STRONGSORT.MAX_UNMATCHED_PREDS,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

        )
        return strongsort
    
    elif tracker_type == 'ocsort':
        ocsort = OCSort(
            det_thresh=cfg.OCSORT.DET_THRESH,
            max_age=cfg.OCSORT.MAX_AGE,
            min_hits=cfg.OCSORT.MIN_HITS,
            iou_threshold=cfg.OCSORT.IOU_THRESH,
            delta_t=cfg.OCSORT.DELTA_T,
            asso_func=cfg.OCSORT.ASSO_FUNC,
            inertia=cfg.OCSORT.INERTIA,
            use_byte=cfg.OCSORT.USE_BYTE,
        )
        return ocsort
    
    elif tracker_type == 'bytetrack':
        bytetracker = BYTETracker(
            track_thresh=cfg.BYTETRACK.TRACK_TRESH,
            match_thresh=cfg.BYTETRACK.MATCH_THRESH,
            track_buffer=cfg.BYTETRACK.TRACK_BUFFER,
            frame_rate=cfg.BYTETRACK.FRAME_RATE
        )
        return bytetracker
    else:
        print('No such tracker')
        exit()