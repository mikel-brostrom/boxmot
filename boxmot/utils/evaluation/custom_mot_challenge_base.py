import numpy as np

from trackeval.utils import TrackEvalException


class CustomMotChallengeBase:
    """Shared helpers for multiclass MOT TrackEval dataset adapters."""

    @staticmethod
    def _normalize_class_config(cfg, default_classes):
        real_classes = [cls.lower() for cls in cfg.get("CLASSES_TO_EVAL", default_classes)]
        class_ids = cfg.get("CLASS_IDS")
        if class_ids is not None:
            class_ids = [int(cid) for cid in class_ids]
        return real_classes, class_ids

    def _configure_class_data(
        self,
        real_classes,
        class_ids,
        default_class_name_to_id,
        *,
        validate_against_default,
        invalid_class_message,
    ):
        if class_ids is not None:
            if len(class_ids) != len(real_classes):
                raise TrackEvalException("CLASS_IDS must have the same length as CLASSES_TO_EVAL.")
            self.valid_classes = real_classes
            self.class_list = real_classes
            self.class_name_to_class_id = {cls: int(cid) for cls, cid in zip(real_classes, class_ids)}
        else:
            self.class_name_to_class_id = default_class_name_to_id.copy()
            if validate_against_default:
                self.valid_classes = list(default_class_name_to_id)
                self.class_list = [cls if cls in self.valid_classes else None for cls in real_classes]
                if not all(self.class_list):
                    raise TrackEvalException(invalid_class_message)
            else:
                self.valid_classes = real_classes
                self.class_list = real_classes

        self.valid_class_numbers = list(self.class_name_to_class_id.values())

    @staticmethod
    def _filter_super_categories(class_list, super_categories):
        filtered = {
            name: [cls for cls in members if cls in class_list]
            for name, members in super_categories.items()
        }
        return {name: members for name, members in filtered.items() if members}

    @staticmethod
    def _relabel_track_ids(data, num_timesteps):
        for source_key, target_key in (("gt_ids", "num_gt_ids"), ("tracker_ids", "num_tracker_ids")):
            unique_ids = []
            for timestep in range(num_timesteps):
                unique_ids += list(np.unique(data[source_key][timestep]))

            if unique_ids:
                unique_ids = np.unique(unique_ids)
                id_map = np.nan * np.ones((np.max(unique_ids) + 1))
                id_map[unique_ids] = np.arange(len(unique_ids))
                for timestep in range(num_timesteps):
                    if len(data[source_key][timestep]) > 0:
                        data[source_key][timestep] = id_map[data[source_key][timestep]].astype(int)
                data[target_key] = len(unique_ids)
            else:
                data[target_key] = 0

