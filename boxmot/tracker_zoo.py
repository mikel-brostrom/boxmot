# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from types import SimpleNamespace

import yaml

from boxmot.utils import BOXMOT


def get_tracker_config(tracker_type):
    tracking_config = \
        BOXMOT /\
        'configs' /\
        (tracker_type + '.yaml')
    return tracking_config


def create_tracker(tracker_type, tracker_config=None, reid_weights=None, device=None, half=None, per_class=None, evolve_param_dict=None):
    # If config_dict is not provided, read from the file
    if evolve_param_dict is None:
        with open(tracker_config, "r") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg = SimpleNamespace(**cfg)  # easier dict access by dot, instead of ['']
    else:
        print('passing this strongsort config', evolve_param_dict)
        cfg = SimpleNamespace(**evolve_param_dict)  # Use provided dict

    if tracker_type == 'strongsort':
        from boxmot.trackers.strongsort.strong_sort import StrongSORT
        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.max_dist,
            max_iou_dist=cfg.max_iou_dist,
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            nn_budget=cfg.nn_budget,
            mc_lambda=cfg.mc_lambda,
            ema_alpha=cfg.ema_alpha,

        )
        return strongsort

    elif tracker_type == 'ocsort':
        from boxmot.trackers.ocsort.ocsort import OCSort
        print(cfg)
        ocsort = OCSort(
            per_class=per_class,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            asso_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
            use_byte=cfg.use_byte,
            Q_xy_scaling=cfg.Q_xy_scaling,
            Q_s_scaling=cfg.Q_s_scaling
        )
        return ocsort

    elif tracker_type == 'bytetrack':
        from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(
            per_class=per_class,
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            frame_rate=cfg.frame_rate
        )
        return bytetracker

    elif tracker_type == 'botsort':
        from boxmot.trackers.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            per_class=per_class,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            cmc_method=cfg.cmc_method,
            frame_rate=cfg.frame_rate,
        )
        return botsort
    elif tracker_type == 'deepocsort':
        from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort

        deepocsort = DeepOCSort(
            reid_weights,
            device,
            half,
            per_class=per_class,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
            Q_xy_scaling=cfg.Q_xy_scaling,
            Q_s_scaling=cfg.Q_s_scaling
        )
        return deepocsort
    elif tracker_type == 'hybridsort':
        from boxmot.trackers.hybridsort.hybridsort import HybridSORT

        hybridsort = HybridSORT(
            reid_weights,
            device,
            half,
            per_class=per_class,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
            longterm_reid_weight=cfg.longterm_reid_weight,
            TCM_first_step_weight=cfg.TCM_first_step_weight,
            use_byte=cfg.use_byte,
        )
        return hybridsort
    elif tracker_type == 'imprassoc':
        from boxmot.trackers.imprassoc.impr_assoc_tracker import ImprAssocTrack
        imprassoc = ImprAssocTrack(
            reid_weights,
            device,
            half,
            per_class=per_class,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            second_match_thresh=cfg.second_match_thresh,
            overlap_thresh=cfg.overlap_thresh,
            lambda_=cfg.lambda_,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            cmc_method=cfg.cmc_method,
            frame_rate=cfg.frame_rate,
        )
        return imprassoc
    else:
        print('No such tracker')
        exit()
