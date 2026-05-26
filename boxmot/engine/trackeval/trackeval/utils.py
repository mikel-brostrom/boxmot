
import os
import csv
import argparse
from collections import OrderedDict


def init_config(config, default_config, name=None):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if name and config['PRINT_CONFIG']:
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config


def update_config(config):
    """
    Parse the arguments of a script and updates the config values for a given value if specified in the arguments.
    :param config: the config to update
    :return: the updated config
    """
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    return config


def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def validate_metrics_list(metrics_list):
    """Get names of metric class and ensures they are unique, further checks that the fields within each metric class
    do not have overlapping names.
    """
    metric_names = [metric.get_name() for metric in metrics_list]
    # check metric names are unique
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException('Code being run with multiple metrics of the same name')
    fields = []
    for m in metrics_list:
        fields += m.fields
    # check metric fields are unique
    if len(fields) != len(set(fields)):
        raise TrackEvalException('Code being run with multiple metrics with fields of the same name')
    return metric_names


def write_summary_results(summaries, cls, output_folder):
    """Write summary results to file"""

    fields = sum([list(s.keys()) for s in summaries], [])
    values = sum([list(s.values()) for s in summaries], [])

    # In order to remain consistent upon new fields being adding, for each of the following fields if they are present
    # they will be output in the summary first in the order below. Any further fields will be output in the order each
    # metric family is called, and within each family either in the order they were added to the dict (python >= 3.6) or
    # randomly (python < 3.6).
    default_order = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)',
                     'HOTALocA(0)', 'MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'CLR_TP', 'CLR_FN',
                     'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'sMOTA', 'IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP',
                     'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
    default_ordered_dict = OrderedDict(zip(default_order, [None for _ in default_order]))
    for f, v in zip(fields, values):
        default_ordered_dict[f] = v
    for df in default_order:
        if default_ordered_dict[df] is None:
            del default_ordered_dict[df]
    fields = list(default_ordered_dict.keys())
    values = list(default_ordered_dict.values())

    out_file = os.path.join(output_folder, cls + '_summary.txt')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(fields)
        writer.writerow(values)


def write_summary_results_multicls(summaries_dict, cls_list, output_folder):
    """Write summary results to file"""

    out_file = os.path.join(output_folder,  'all_cls_summary.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # summaries = summaries_dict[cls_list[0]]
    for index, cls in enumerate(cls_list):
        summaries = summaries_dict[cls]

        fields = sum([list(s.keys()) for s in summaries], [])
        values = sum([list(s.values()) for s in summaries], [])

        # In order to remain consistent upon new fields being adding, for each of the following fields if they are present
        # they will be output in the summary first in the order below. Any further fields will be output in the order each
        # metric family is called, and within each family either in the order they were added to the dict (python >= 3.6) or
        # randomly (python < 3.6).
        default_order = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)',
                        'HOTALocA(0)', 'MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'CLR_TP', 'CLR_FN',
                        'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'sMOTA', 'IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP',
                        'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
        default_ordered_dict = OrderedDict(zip(default_order, [None for _ in default_order]))
        for f, v in zip(fields, values):
            default_ordered_dict[f] = v
        for df in default_order:
            if default_ordered_dict[df] is None:
                del default_ordered_dict[df]
        fields = list(default_ordered_dict.keys())
        values = list(default_ordered_dict.values())
        fields.insert(0, 'cls')
        values.insert(0, cls)

        if index == 0:
            with open(out_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerow(values)
        else:
            with open(out_file, 'a', newline='') as f:
                writer = csv.writer(f)
                # writer.writerow(fields)
                writer.writerow(values)

def write_summary_results_all_seq(seq_summaries, cls_list, output_folder):
    """Write summary results to file"""

    out_file = os.path.join(output_folder,  'all_seq_summary.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    seq_list = list(seq_summaries.keys())
    seq_list.sort()

    # write head
    default_order = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)',
                'HOTALocA(0)', 'MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'CLR_TP', 'CLR_FN',
                'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'sMOTA', 'IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP',
                'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
    default_ordered_dict = OrderedDict(zip(default_order, [None for _ in default_order]))
    summaries_first = seq_summaries[seq_list[0]][cls_list[0]]
    fields = sum([list(s.keys()) for s in summaries_first], [])
    values = sum([list(s.values()) for s in summaries_first], [])   
    for f, v in zip(fields, values):
        default_ordered_dict[f] = v
    for df in default_order:
        if default_ordered_dict[df] is None:
            del default_ordered_dict[df]
    final_fields = list(default_ordered_dict.keys())
    final_fields.insert(0, 'cls')
    final_fields.insert(0, 'seq')

    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(final_fields)

    for seq in seq_list:
        for cls in cls_list:
            summaries = seq_summaries[seq][cls]

            fields = sum([list(s.keys()) for s in summaries], [])
            values = sum([list(s.values()) for s in summaries], [])        

            order_dict = OrderedDict(zip(final_fields, [None for _ in final_fields]))
            for f, v in zip(fields, values):
                order_dict[f] = v
            order_dict['cls'] = cls
            order_dict['seq'] = seq

            for df in final_fields:
                if order_dict[df] is None:
                    order_dict[df] = -10000
            values = list(order_dict.values())

            with open(out_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(values)

def write_detailed_results(details, cls, output_folder):
    """Write detailed results to file"""
    sequences = details[0].keys()
    fields = ['seq'] + sum([list(s['COMBINED_SEQ'].keys()) for s in details], [])
    out_file = os.path.join(output_folder, cls + '_detailed.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for seq in sorted(sequences):
            if seq == 'COMBINED_SEQ':
                continue
            writer.writerow([seq] + sum([list(s[seq].values()) for s in details], []))
        writer.writerow(['COMBINED'] + sum([list(s['COMBINED_SEQ'].values()) for s in details], []))


def load_detail(file):
    """Loads detailed data for a tracker."""
    data = {}
    with open(file) as f:
        for i, row_text in enumerate(f):
            row = row_text.replace('\r', '').replace('\n', '').split(',')
            if i == 0:
                keys = row[1:]
                continue
            current_values = row[1:]
            seq = row[0]
            if seq == 'COMBINED':
                seq = 'COMBINED_SEQ'
            if (len(current_values) == len(keys)) and seq != '':
                data[seq] = {}
                for key, value in zip(keys, current_values):
                    data[seq][key] = float(value)
    return data


class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""
    ...
