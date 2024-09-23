# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license



from types import SimpleNamespace
import yaml
from boxmot.utils import BOXMOT

def get_tracker_config(tracker_type):
    tracking_config = BOXMOT / 'configs' / (tracker_type + '.yaml')
    return tracking_config

def create_tracker(tracker_type, tracker_config=None, reid_weights=None, device=None, half=None, per_class=None, evolve_param_dict=None):
    # Load the configuration from file or use the provided dictionary
    if evolve_param_dict is None:
        with open(tracker_config, "r") as f:
            tracker_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        tracker_args = evolve_param_dict

    reid_args = {
        'reid_weights': reid_weights,
        'device': device,
        'half': half,
        'per_class': per_class
    }

    if tracker_type == 'strongsort':
        from boxmot.trackers.strongsort.strong_sort import StrongSORT
        tracker_args.update(reid_args)
        tracker_args.pop('per_class')
        return StrongSORT(**tracker_args)

    elif tracker_type == 'ocsort':
        from boxmot.trackers.ocsort.ocsort import OCSort
        return OCSort(**tracker_args)

    elif tracker_type == 'bytetrack':
        from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
        return BYTETracker(**tracker_args)

    elif tracker_type == 'botsort':
        from boxmot.trackers.botsort.bot_sort import BoTSORT
        tracker_args.update(reid_args)
        return BoTSORT(**tracker_args)

    elif tracker_type == 'deepocsort':
        from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort
        tracker_args.update(reid_args)
        return DeepOCSort(**tracker_args)

    elif tracker_type == 'hybridsort':
        from boxmot.trackers.hybridsort.hybridsort import HybridSORT
        tracker_args.update(reid_args)
        return HybridSORT(**tracker_args)

    elif tracker_type == 'imprassoc':
        from boxmot.trackers.imprassoc.impr_assoc_tracker import ImprAssocTrack
        tracker_args.update(reid_args)
        return ImprAssocTrack(**tracker_args)

    else:
        print('No such tracker')
        exit()