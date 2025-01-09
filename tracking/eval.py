from collections import deque
import subprocess
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from tracking.utils import decrease_mot_dataset_fps

import os
import threading

# ANSI escape codes for cursor movement
UP = "\x1B[3A"
DOWN = "\x1B[3B"
CLR = "\x1B[0K"

# ANSI escape codes for colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

bar_format = (
    f'{GREEN}{{desc}}{RESET}: '
    f'{GREEN}{{percentage:3.0f}}%{RESET} |'
    f'{{bar}}{RESET}| '
    f'{GREEN}{{n_fmt}}/{{total_fmt}}{RESET} '
    f'[{CYAN}{{elapsed}}<{{remaining}}{RESET}, '
    f'{CYAN}{{rate_fmt}}{RESET}]'
)

EVAL_DIR = Path('runs/eval')
PLOTS_DIRNAME = 'plots'
RESULTS_DIRNAME = 'results'
RESULTS_FILENAME = 'results.json'

EVAL_DIR.mkdir(exist_ok=True, parents=True)

# In case of FPS check
FPS_DEFAULT_SET = ['5', '10', '15', '20']


def run_val_script(yolo_model, tracking_method, reid_model, source, val_args):
    command = [
        'python', 'tracking/val.py',
        '--yolo-model', yolo_model,
        '--tracking-method', tracking_method,
        '--reid-model', reid_model,
        '--source', source,
        "--ci"
    ]

    command.extend(val_args)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
    )

    output_lines = []
    lock = threading.Lock()

    def process_stream(stream):
        try:
            for line in iter(stream.readline, ''):
                line = line.strip()
                if line:
                    with lock:
                        if line.startswith('Frames:'):
                            print(f'{UP}{line}{CLR}\n{CLR}\n', flush=True)
                        else:
                            # For regular output, use tqdm.write
                            output_lines.append(line)
                            if len(output_lines) > 10:
                                output_lines.pop(0)
                            tqdm.write(line)
        except Exception as e:
            tqdm.write(f"Error processing stream: {e}")

    stdout_thread = threading.Thread(
        target=process_stream, args=(process.stdout,))
    stderr_thread = threading.Thread(
        target=process_stream, args=(process.stderr,))

    stdout_thread.daemon = True
    stderr_thread.daemon = True

    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to complete
    return_code = process.wait()

    # Wait for the output threads to finish
    stdout_thread.join()
    stderr_thread.join()

    # Add a newline after progress bar completes
    print(f'{UP}{CLR}')

    # Close the pipes
    process.stdout.close()
    process.stderr.close()

    if return_code != 0:
        return "\n".join(output_lines[-10:]), "\n".join(output_lines[-20:])

    return "\n".join(output_lines[-10:]), None


def parse_output(output):
    # Extract the last line which contains the metrics
    lines = output.strip().split('\n')
    if lines:
        last_line = lines[-1]
        try:
            metrics = json.loads(last_line)
            return metrics
        except json.JSONDecodeError:
            print(f"Failed to parse output: {last_line}")
    return None


def fps_sort_key(value):
    if value == "original":
        return float('inf')  # Treat "original" as the largest value
    return int(value)  # Convert numerical strings to integers for sorting


def plot_results(results,
                 ds_name: str):
    for yolo_model, reid_models in results.items():
        for reid_model, tracking_methods in reid_models.items():
            plt.figure()
            for tracking_method, fps_metrics in tracking_methods.items():
                fps_values = sorted(list(fps_metrics.keys()), key=fps_sort_key)
                hota_values = [fps_metrics[fps]['HOTA'] for fps in fps_values]

                plt.plot(fps_values, hota_values,
                         label=tracking_method, marker='o')
                for fps, hota in zip(fps_values, hota_values):
                    plt.text(fps, hota, f'{hota:.2f}', ha='right')

            plt.xlabel('FPS')
            plt.ylabel('HOTA')
            plt.title(f'Detector: {yolo_model}, ReID Model: {reid_model}')
            plt.legend(title='Tracking Method')
            plt.grid(True)

            plot_name = f'{ds_name}_{yolo_model.split(".")[0]}_{
                reid_model.split(".")[0]}_plot.png'
            plot_path = EVAL_DIR / ds_name / PLOTS_DIRNAME / plot_name
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(plot_path)
            plt.close()


def load_json(path: Path):
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_json(values: dict, path: Path):
    with open(path, 'w') as f:
        json.dump(values, f, indent=4)


def load_results(ds_name: str, yolo_names: list[str], reid_names: list[str],
                 tracking_methods: list[str], fps_values: list[str]):
    results_dirpath = EVAL_DIR / ds_name / RESULTS_DIRNAME
    results_dirpath.mkdir(parents=True, exist_ok=True)
    results = {}

    yolo_names = [yolo_name.split('.')[0] for yolo_name in yolo_names]
    reid_names = [reid_name.split('.')[0] for reid_name in reid_names]
    tracking_methods = [tracking_method.split('.')[0]
                        for tracking_method in tracking_methods]

    for yolo_path in results_dirpath.iterdir():
        if not yolo_path.is_dir():
            continue

        yolo_name = yolo_path.name
        if yolo_name not in yolo_names:
            continue

        if yolo_name not in results:
            results[yolo_name] = {}

        for reid_path in yolo_path.iterdir():
            if not reid_path.is_dir():
                continue

            reid_name = reid_path.name
            if reid_name not in reid_names:
                continue

            if reid_name not in results[yolo_name]:
                results[yolo_name][reid_name] = {}

            for tracking_path in reid_path.iterdir():
                if not tracking_path.is_dir():
                    continue

                tracking_method = tracking_path.name
                if tracking_method not in tracking_methods:
                    continue

                if tracking_method not in results[yolo_name][reid_name]:
                    results[yolo_name][reid_name][tracking_method] = {}

                for fps_path in tracking_path.iterdir():
                    if not fps_path.is_dir():
                        continue

                    fps_dirname = fps_path.name
                    fps = fps_dirname[:fps_dirname.find('_')]
                    if fps not in fps_values:
                        continue

                    results_path = (results_dirpath / yolo_name / reid_name /
                                    tracking_method / fps_dirname /
                                    RESULTS_FILENAME)

                    results[yolo_name][reid_name][tracking_method][fps] = load_json(
                        results_path)
    return results


def save_metrics(metrics: dict[str, float],
                 ds_name: str, yolo_name: str, reid_name: str,
                 tracking_method: str, fps: str):
    fps_dirname = f"{fps}_FPS"
    results_dirpath = EVAL_DIR / ds_name / RESULTS_DIRNAME
    results_path = (results_dirpath / yolo_name / reid_name /
                    tracking_method / fps_dirname / RESULTS_FILENAME)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(metrics, results_path)


def save_results(
        results: dict[str, dict[str,
                                dict[str,
                                     dict[str,
                                          dict[str, float]]]]],
        ds_name: str):
    for yolo_name in results:
        yolo_dict = results[yolo_name]
        for reid_name in yolo_dict:
            reid_dict = yolo_dict[reid_name]
            for tracking_method in reid_dict:
                tracking_dict = reid_dict[tracking_method]
                for fps in tracking_dict:
                    metrics_values = tracking_dict[fps]
                    save_metrics(metrics_values, ds_name, yolo_name,
                                 reid_name, tracking_method, fps)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trackers with different combinations.',
        allow_abbrev=False)
    parser.add_argument('--yolo-models', nargs='+', required=True,
                        help='List of YOLO models to evaluate.')
    parser.add_argument('--tracking-methods', nargs='+', required=True,
                        help='List of tracking methods to evaluate.')
    parser.add_argument('--reid-models', nargs='+', required=True,
                        help='List of ReID models to evaluate.')
    parser.add_argument('--source', required=True,
                        help='Source data for evaluation.')

    fps_parser = parser.add_mutually_exclusive_group(required=False)
    fps_parser.add_argument('--fps-check', dest='fps_check', action='store_true',
                            help='Check dataset working on different FPS values.')
    fps_parser.add_argument('--no-fps-check', dest='fps_check', action='store_false',
                            help='Do not check dataset working on different FPS values.')
    parser.set_defaults(fps_check=True)

    parser.add_argument('--fps-values', nargs='+',
                        default=FPS_DEFAULT_SET, type=str)

    parser.add_argument('--new', action='store_true',
                        help='Force recalculation of HOTA-MOTA-IDF1.')

    args, unknown_args = parser.parse_known_args()

    fps_values = ['original']
    if args.fps_check:
        fps_values = fps_values + args.fps_values

    source_path = Path(args.source)

    dataset_part = source_path.name
    dataset_path = source_path.parent
    dataset_name = dataset_path.name

    results = load_results(
        dataset_name, args.yolo_models, args.reid_models,
        args.tracking_methods, fps_values) if not args.new else {}

    # Progress bar for YOLO models
    for yolo_model in tqdm(args.yolo_models, desc='Evaluating YOLO models', unit='model', bar_format=bar_format, position=1, leave=False):
        yolo_name = yolo_model.split(".")[0]

        if yolo_name not in results:

            results[yolo_name] = {}

        # Progress bar for ReID models
        for reid_model in tqdm(args.reid_models, desc=f'Evaluating ReID models for {yolo_model}', unit='reid_model', bar_format=bar_format, position=2, leave=False):
            reid_name = reid_model.split(".")[0]
            if reid_name not in results[yolo_name]:
                results[yolo_name][reid_name] = {}

            # Progress bar for tracking methods
            for tracking_method in tqdm(args.tracking_methods, desc=f'Evaluating tracking methods for {yolo_model} with {reid_model}', unit='method', bar_format=bar_format, position=3, leave=False):
                if tracking_method not in results[yolo_name][reid_name]:
                    results[yolo_name][reid_name][tracking_method] = {}

                # Progress bar for FPS values
                for fps in tqdm(fps_values, desc=f'Evaluating FPS for {tracking_method}', unit='fps', bar_format=bar_format, position=4, leave=False):
                    if fps in results[yolo_name][reid_name][tracking_method]:
                        continue

                    cur_source_path = source_path

                    if fps != 'original':

                        new_dataset_path = decrease_mot_dataset_fps(
                            dataset_path, int(fps), replace_if_exists=False)

                        cur_source_path = new_dataset_path / dataset_part

                    output, error = run_val_script(
                        yolo_model, tracking_method, reid_model,
                        str(cur_source_path), unknown_args)

                    if error is not None:
                        raise RuntimeError(
                            f"Error occurred for {yolo_name} with "
                            f"{tracking_method} and {
                                reid_name} at {fps} FPS: "
                            f"{"\n".join(error.split("\n")[-10:])}")

                    metrics = parse_output(output)

                    if metrics:
                        results[yolo_name][reid_name][tracking_method][fps] = metrics
                        # Checkpointing
                        save_results(results, dataset_name)

    print(f"{DOWN}" * 2)
    print(json.dumps(results, indent=4))
    plot_results(results, dataset_name)
    # save_results(results)
    save_results(results, dataset_name)


if __name__ == '__main__':
    main()
