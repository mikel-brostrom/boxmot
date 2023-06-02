from pathlib import Path
import yaml
from types import SimpleNamespace
from boxmot.utils import BOXMOT


def get_tracker_config(tracker_type):
    tracking_config = \
        BOXMOT /\
        tracker_type /\
        'configs' /\
        (tracker_type + '.yaml')
    return tracking_config
    

def create_tracker(tracker_type, tracker_config, reid_weights, device, half):

    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']
    
    if tracker_type == 'strongsort':
        from boxmot.strongsort.strong_sort import StrongSORT
        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.max_dist,
            max_iou_dist=cfg.max_iou_dist,
            max_age=cfg.max_age,
            max_unmatched_preds=cfg.max_unmatched_preds,
            n_init=cfg.n_init,
            nn_budget=cfg.nn_budget,
            mc_lambda=cfg.mc_lambda,
            ema_alpha=cfg.ema_alpha,

        )
        return strongsort
    
    elif tracker_type == 'ocsort':
        from boxmot.ocsort.ocsort import OCSort
        ocsort = OCSort(
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
            use_byte=cfg.use_byte,
        )
        return ocsort
    
    elif tracker_type == 'bytetrack':
        from boxmot.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            frame_rate=cfg.frame_rate
        )
        return bytetracker
    
    elif tracker_type == 'botsort':
        from boxmot.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            track_high_thresh=cfg.track_high_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer =cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            cmc_method =cfg.cmc_method,
            frame_rate=cfg.frame_rate,
            lambda_=cfg.lambda_
        )
        return botsort
    elif tracker_type == 'deepocsort':
        from boxmot.deepocsort.ocsort import OCSort
        deepocsort = OCSort(
            reid_weights,
            device,
            half,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
        )
        return deepocsort
    else:
        print('No such tracker')
        exit()