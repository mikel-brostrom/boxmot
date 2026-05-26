import json
import argparse
from .format_converter import GroundTruthBURSTFormatToTAOFormatConverter, PredictionBURSTFormatToTAOFormatConverter


def main(args):
    with open(args.gt_input_file) as f:
        ali_format_gt = json.load(f)
    tao_format_gt = GroundTruthBURSTFormatToTAOFormatConverter(
        ali_format_gt, args.split).convert()
    with open(args.gt_output_file, 'w') as f:
        json.dump(tao_format_gt, f)

    if args.pred_input_file is None:
        return
    with open(args.pred_input_file) as f:
        ali_format_pred = json.load(f)
    tao_format_pred = PredictionBURSTFormatToTAOFormatConverter(
        tao_format_gt, ali_format_pred, args.split,
        args.exemplar_guided).convert()
    with open(args.pred_output_file, 'w') as f:
        json.dump(tao_format_pred, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_input_file', type=str,
        default='../data/gt/tsunami/exemplar_guided/validation_all_annotations.json')
    parser.add_argument('--gt_output_file', type=str,
                        default='/tmp/val_gt.json')
    parser.add_argument('--pred_input_file', type=str,
                        default='../data/trackers/tsunami/exemplar_guided/STCN_off_the_shelf/data/results.json')
    parser.add_argument('--pred_output_file', type=str,
                        default='/tmp/pred.json')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--exemplar_guided', type=bool, default=True)
    args_ = parser.parse_args()
    main(args_)
