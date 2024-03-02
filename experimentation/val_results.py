import re
import sys
import argparse
import subprocess
from boxmot.utils import EXAMPLES, ROOT, WEIGHTS, EXPERIMENTATION
from pathlib import Path
from experimentation.utils import (
    download_mot_eval_tools,
    download_mot_dataset,
    unzip_mot_dataset,
    eval_setup
)
from ultralytics.utils.files import increment_path 



def parse_mot_results(results):
        """Extract the COMBINED HOTA, MOTA, IDF1 from the results generate by the
           run_mot_challenge.py script.

        Args:
            str: mot_results

        Returns:
            (dict): {'HOTA': x, 'MOTA':y, 'IDF1':z}
        """
        combined_results = results.split('COMBINED')[2:-1]
        # robust way of getting first ints/float in string
        combined_results = [float(re.findall("[-+]?(?:\d*\.*\d+)", f)[0]) for f in combined_results]
        # pack everything in dict
        combined_results = {key: value for key, value in zip(['HOTA', 'MOTA', 'IDF1'], combined_results)}
        return combined_results


def trackeval(
    args,
    seq_paths,
    save_dir,
    MOT_results_folder,
    gt_folder,
    metrics = ["HOTA", "CLEAR", "Identity"]
):
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.
    
    Parameters:
        script_path (str): The path to the evaluation script to run.
        trackers_folder (str): The folder where tracker results are stored.
        metrics (list): A list of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].
        num_parallel_cores (int): The number of parallel cores to use for evaluation. Defaults to 4.
    
    Outputs:
        Prints the standard output and standard error from the evaluation script.
    """
    # Define paths
    d = [seq_path.parent.name for seq_path in seq_paths]
    # Prepare arguments for subprocess call
    print('args.exp_folder_path', args.exp_folder_path) 
    args = [
        sys.executable, EXPERIMENTATION / 'val_utils' / 'scripts' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", "",
        "--TRACKERS_FOLDER", args.exp_folder_path,
        "--TRACKERS_TO_EVAL", "",
        "--SPLIT_TO_EVAL", "train",
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--SEQ_INFO", *d
    ]

    # Execute the evaluation script
    p = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    # Output the results
    print("Standard Output:\n", stdout)
    if stderr:
        print("Standard Error:\n", stderr)
    return stdout


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=0,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'mot',
                        help='save results to project/name')
    parser.add_argument('--name', default='yolov8n_osnet_x0_25_msmt17',
                        help='save results to project/name')
    parser.add_argument('--dets', type=str, default='yolov8n',
                        help='the folder name under project to load the detections from')
    parser.add_argument('--embs', type=str, default='osnet_x0_25_msmt17',
                        help='the folder name under project/dets to load the embeddings from')
    parser.add_argument('--exist-ok', action='store_true', default=True,
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--benchmark', type=str, default='MOT17',
                        help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str, default='train',
                        help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    return opt


def run_trackeval(opt):
    if opt is None:
        opt = parse_opt()
        exp_folder_path = opt.project / (str(opt.dets) + "_" + str(opt.embs))
        exp_folder_path = increment_path(path=exp_folder_path, sep="_", exist_ok=opt.exist_ok)
        opt.exp_folder_path = exp_folder_path
    else:
        opt = opt
        opt.exist_ok = False

    val_tools_path = EXPERIMENTATION / 'val_utils'
    download_mot_eval_tools(val_tools_path)
    zip_path = download_mot_dataset(val_tools_path, opt.benchmark)
    unzip_mot_dataset(zip_path, val_tools_path, opt.benchmark)
    seq_paths, save_dir, MOT_results_folder, gt_folder = eval_setup(opt, val_tools_path)
    results = trackeval(opt, seq_paths, save_dir, MOT_results_folder, gt_folder)
    combined_results = parse_mot_results(results)
    print(combined_results)
    return combined_results

if __name__ == "__main__":
    run_trackeval(None)

