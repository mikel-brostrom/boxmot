from trackers.strongsort.utils.parser import get_config
from trackers.strongsort.strong_sort import StrongSORT
from trackers.ocsort.ocsort import OCSort
from trackers.bytetrack.byte_tracker import BYTETracker
from trackers.botsort.bot_sort import BoTSORT

def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    
    cfg = get_config()
    cfg.merge_from_file(tracker_config)
    
    if tracker_type == 'strongsort':

        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.strongsort.max_dist,
            max_iou_dist=cfg.strongsort.max_iou_dist,
            max_age=cfg.strongsort.max_age,
            max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
            n_init=cfg.strongsort.n_init,
            nn_budget=cfg.strongsort.nn_budget,
            mc_lambda=cfg.strongsort.mc_lambda,
            ema_alpha=cfg.strongsort.ema_alpha,

        )
        return strongsort
    
    elif tracker_type == 'ocsort':
        ocsort = OCSort(
            det_thresh=cfg.ocsort.det_thresh,
            max_age=cfg.ocsort.max_age,
            min_hits=cfg.ocsort.min_hits,
            iou_threshold=cfg.ocsort.iou_thresh,
            delta_t=cfg.ocsort.delta_t,
            asso_func=cfg.ocsort.asso_func,
            inertia=cfg.ocsort.inertia,
            use_byte=cfg.ocsort.use_byte,
        )
        return ocsort
    
    elif tracker_type == 'bytetrack':
        bytetracker = BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=cfg.bytetrack.frame_rate
        )
        return bytetracker
    
    elif tracker_type == 'botsort':
        botsort = BoTSORT(
            appearance_descriptor_weights,
            device,
            half,
            track_high_thresh = 0.45,
            new_track_thresh= 0.6,
            track_buffer = 30,
            match_thresh= 0.8,
            proximity_thresh= 0.5,
            appearance_thresh= 0.25,
            cmc_method = 'sparseOptFlow',
            frame_rate=30
        )
        return botsort
    else:
        print('No such tracker')
        exit()