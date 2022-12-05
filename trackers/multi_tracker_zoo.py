from trackers.strong_sort.utils.parser import get_config
from trackers.strong_sort.strong_sort import StrongSORT
from trackers.ocsort.ocsort import OCSort
from trackers.bytetrack.byte_tracker import BYTETracker
from trackers.strong_ocsort.strong_ocsort import StrongOCSort


def create_tracker(tracker_type, appearance_descriptor_weights, device, half):
    if tracker_type == 'strongsort':
        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file('trackers/strong_sort/configs/strong_sort.yaml')

        strongsort = StrongSORT(
            appearance_descriptor_weights,
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
            det_thresh=0.45,
            iou_threshold=0.2,
            use_byte=False 
        )
        return ocsort
    elif tracker_type == 'bytetrack':
        bytetracker = BYTETracker(
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )
        return bytetracker
    elif tracker_type == 'strong-ocsort':
        # initialize Strong-OCSort
        cfg = get_config()
        cfg.merge_from_file('trackers/strong_ocsort/configs/strong_ocsort.yaml')

        strongocsort = StrongOCSort(
            appearance_descriptor_weights,
            device,
            half,
            det_thresh=cfg.STRONG_OCSORT.DET_THRESH,
            max_dist=cfg.STRONG_OCSORT.MAX_DIST,
            nn_budget=cfg.STRONG_OCSORT.NN_BUDGET,
            ema_alpha=cfg.STRONG_OCSORT.EMA_ALPHA,
            max_age=cfg.STRONG_OCSORT.MAX_AGE,
            min_hits=cfg.STRONG_OCSORT.MIN_HITS,
            iou_threshold=cfg.STRONG_OCSORT.IOU_THRESHOLD,
            delta_t=cfg.STRONG_OCSORT.DELTA_T,
            inertia=cfg.STRONG_OCSORT.INERTIA,
            use_byte=cfg.STRONG_OCSORT.USE_BYTE,
            use_resurrection=cfg.STRONG_OCSORT.USE_RESURRECTION,
        )

        return strongocsort
    else:
        print('No such tracker')
        exit()