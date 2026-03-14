"""Run OBB MOT TrackEval with BoxMOT's multiclass-compatible dataset adapter."""

from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path


current_dir = Path(__file__).resolve().parent
boxmot_dir = current_dir.parent
trackeval_dir = boxmot_dir / "engine" / "trackeval"

sys.path.insert(0, str(trackeval_dir))

import trackeval  # noqa: E402

from boxmot.utils.custom_mot_challenge_obb import CustomMotChallengeOBB


if __name__ == "__main__":
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = False
    default_dataset_config = CustomMotChallengeOBB.get_default_dataset_config()
    default_metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"]}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if setting == "CLASS_IDS":
            parser.add_argument("--" + setting, nargs="+")
        elif isinstance(config[setting], list) or config[setting] is None:
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)

    args = parser.parse_args().__dict__
    for setting, value in args.items():
        if value is None:
            continue

        if isinstance(config[setting], bool):
            if value == "True":
                parsed_value = True
            elif value == "False":
                parsed_value = False
            else:
                raise Exception("Command line parameter " + setting + " must be True or False")
        elif isinstance(config[setting], int):
            parsed_value = int(value)
        elif setting == "CLASS_IDS":
            parsed_value = [int(v) for v in value]
        else:
            parsed_value = value

        config[setting] = parsed_value

    eval_config = {k: v for k, v in config.items() if k in default_eval_config}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config}
    metrics_config["METRICS"] = [metric.lower() for metric in metrics_config["METRICS"]]

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [CustomMotChallengeOBB(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name().lower() in metrics_config["METRICS"]:
            metrics_list.append(metric())
    if not metrics_list:
        raise Exception("No metrics selected for evaluation")

    evaluator.evaluate(dataset_list, metrics_list)
