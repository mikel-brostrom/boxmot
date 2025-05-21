# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from functools import partial
from pathlib import Path
import time
import json
import numpy as np
import logging
import shutil # Added for directory removal
import os # Add os import

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Add this line

import cv2
import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model)
from tracking.relate import compute_relationships
from tracking import vectorize

checker = RequirementsChecker()
checker.check_packages((
    'ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',
    'qdrant-client',  # Added qdrant-client
    'transformers',     # Added transformers
    'torch',            # Added torch
    'Pillow'            # Added Pillow for image processing
))

from ultralytics import YOLO


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.
    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only models do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    # Configure basic logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Clear previous run data if flag is set
    if args.clear_prev_runs:
        exp_dir_to_clear = Path(args.project) / args.name
        if exp_dir_to_clear.exists() and exp_dir_to_clear.is_dir():
            logging.info(f"--clear-prev-runs flag set. Deleting directory: {exp_dir_to_clear}")
            try:
                shutil.rmtree(exp_dir_to_clear)
                logging.info(f"Successfully deleted {exp_dir_to_clear}.")
            except Exception as e:
                logging.error(f"Error deleting {exp_dir_to_clear}: {e}. Please check permissions or manually delete.")
        else:
            logging.info(f"--clear-prev-runs flag set, but directory {exp_dir_to_clear} does not exist or is not a directory. Nothing to delete.")

    # Reinitialize Qdrant collections at the start of the run
    try:
        vectorize.reinitialize_collections()
        logging.info("Successfully reinitialized Qdrant collections.")
    except Exception as e:
        logging.error(f"Failed to reinitialize Qdrant collections: {e}. Terminating.")
        return # or raise e, depending on how critical Qdrant is

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width,
        save_crop=args.save_id_crops
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if not is_ultralytics_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback(
                "on_predict_batch_start",
                lambda p: yolo_model.update_im_paths(p)
            )
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    # store custom args in predictor
    yolo.predictor.custom_args = args

    dataset_frames = []
    exp_dir = Path(yolo.predictor.save_dir)  # This is the actual exp folder used for outputs

    total_start_time = time.time() # Start total timer
    frame_speeds = [] # List to store detailed speed for each frame

    # Variables for selective frame embedding
    last_stored_frame_embedding = None
    last_stored_frame_vector_id = None
    previous_entity_count = -1  # Use -1 to indicate no previous frame data for entity count
    is_first_frame = True

    # Initialize new metrics counters
    num_unique_frames_embedded = 0
    num_frames_skipped_due_to_similarity = 0
    num_entity_detections_processed = 0
    num_entity_vectors_upserted = 0

    for frame_idx, r in enumerate(results):
        # --- Make a copy of the original frame for data operations ---
        clean_frame = r.orig_img.copy()

        # --- Use annotated image for display ---
        # The plot_results modifies r.orig_img in place.
        # We use clean_frame for data, so r.orig_img can be used for plotting.
        img_for_display = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
        
        frame_vector_id_for_json = None
        # current_frame_actual_embedding = None # Will be defined if args.save_dataset

        if args.save_dataset: # Logic for frame embedding is tied to save_dataset
            try:
                # Always generate current frame's embedding using the clean frame
                current_frame_actual_embedding = vectorize.embed_image(clean_frame) # MODIFIED
                current_entity_count = len(r.boxes) if r.boxes is not None and hasattr(r.boxes, 'id') and r.boxes.id is not None else 0

                should_store_new_frame_embedding = True # Default to storing

                if not is_first_frame and last_stored_frame_embedding is not None:
                    similarity = vectorize.calculate_cosine_similarity(
                        current_frame_actual_embedding,
                        last_stored_frame_embedding
                    )
                    if similarity >= args.frame_similarity_threshold and current_entity_count == previous_entity_count:
                        frame_vector_id_for_json = last_stored_frame_vector_id # Reuse ID
                        should_store_new_frame_embedding = False
                        logging.info(f"Frame {frame_idx}: Reusing frame vector ID {frame_vector_id_for_json}. Similarity: {similarity:.4f}, Entities: {current_entity_count}")
                        if not is_first_frame: # Don't count skip for the conceptual first frame comparison if it were to happen
                            num_frames_skipped_due_to_similarity += 1
                if should_store_new_frame_embedding:
                    new_frame_vector_id = vectorize.add_frame_embedding(
                        embedding=current_frame_actual_embedding,
                        frame_idx=frame_idx,
                        timestamp=frame_idx, # Using frame_idx as timestamp for now
                        image=clean_frame
                    )
                    frame_vector_id_for_json = new_frame_vector_id
                    last_stored_frame_embedding = current_frame_actual_embedding
                    last_stored_frame_vector_id = new_frame_vector_id
                    num_unique_frames_embedded += 1
                    if not is_first_frame:
                        logging.info(f"Frame {frame_idx}: Stored new frame vector ID {frame_vector_id_for_json}. Entities: {current_entity_count}")
                    else:
                        logging.info(f"Frame {frame_idx}: Stored new frame vector ID {frame_vector_id_for_json} (first frame). Entities: {current_entity_count}")
                
                previous_entity_count = current_entity_count # Update for the next frame's comparison
                if is_first_frame:
                    is_first_frame = False

            except Exception as e:
                logging.error(f"Error processing frame {frame_idx} for embedding: {e}")
                # Decide if we should assign a None or skip, or use a fallback
                frame_vector_id_for_json = None # Or some error indicator

        if args.save_dataset:
            frame_info = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx,  # Could be replaced with actual timestamp if available
                'geo': None,  # Placeholder for geo location
                'frame_vector_id': frame_vector_id_for_json, # Store the frame vector ID
                'entities': [],
                'relationships': []
            }
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().tolist() if hasattr(r.boxes.xyxy, 'cpu') else r.boxes.xyxy.tolist()
                ids = r.boxes.id.cpu().numpy().tolist() if r.boxes.id is not None and hasattr(r.boxes.id, 'cpu') else (r.boxes.id.tolist() if r.boxes.id is not None else [None]*len(boxes))
                confs = r.boxes.conf.cpu().numpy().tolist() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf.tolist()
                clss = r.boxes.cls.cpu().numpy().tolist() if hasattr(r.boxes.cls, 'cpu') else r.boxes.cls.tolist()
                names = r.names if hasattr(r, 'names') else {}
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    # --- Crop from the clean_frame ---
                    crop = clean_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # MODIFIED
                    tracker_id = ids[i] if ids else None
                    if tracker_id is None:
                        logging.warning(f"Skipping detection with no tracker ID at frame {frame_idx}, detection {i}")
                        continue
                    num_entity_detections_processed += 1
                    entity_id = str(tracker_id)
                    class_idx = None  # Initialize to None to avoid UnboundLocalError
                    class_name = None  # Initialize to None to avoid UnboundLocalError
                    emb = vectorize.embed_crop(crop)
                    
                    # Search for an existing logical entity based on the current detection's embedding
                    found_logical_id, _, payload_from_search, _ = vectorize.search_entity(emb, threshold=0.95)

                    # Determine the logical ID for this entity in the current frame.
                    # entity_id is already str(tracker_id) from earlier in the loop.
                    # class_idx and class_name are initialized to None.
                    if found_logical_id is not None:
                        entity_id = found_logical_id  # Use the matched logical ID
                        if payload_from_search:
                            # Update class info from payload if a match was found
                            class_idx = payload_from_search.get('class', class_idx)
                            class_name = payload_from_search.get('class_name', class_name)
                    # If not found, entity_id remains str(tracker_id), and class_idx/name remain as initialized.

                    # Prepare metadata for Qdrant.
                    # Prioritize current detection's class/name. If not available, use class_idx/name
                    # (which would be from payload if found, or None if new and no detection class).
                    current_detection_class = clss[i] if clss and i < len(clss) else None
                    current_detection_class_name = names.get(int(clss[i]), None) if names and clss and i < len(clss) and clss[i] is not None and int(clss[i]) in names else None

                    meta_class = current_detection_class if current_detection_class is not None else class_idx
                    meta_class_name = current_detection_class_name if current_detection_class_name is not None else class_name
                    
                    entity_metadata = {
                        'bbox': bbox,
                        'id': entity_id,  # This is the logical ID (tracker_id or found_logical_id)
                        'class': meta_class,
                        'class_name': meta_class_name,
                        'confidence': confs[i] if confs and i < len(confs) else None
                    }
                    
                    # Always call add_entity to get/update Qdrant vector and obtain its ID.
                    # vectorize.add_entity is expected to handle upsert based on the logical 'entity_id'.
                    qdrant_vector_id = vectorize.add_entity(
                        emb, 
                        crop, 
                        entity_metadata, 
                        entity_id=entity_id
                    )
                    
                    # Assuming add_entity is successful if no exception, and it always tries an upsert.
                    # A more robust check might involve inspecting the result of qdrant.upsert if available from add_entity
                    num_entity_vectors_upserted += 1
                    
                    # Ensure class_idx and class_name reflect what was actually used in metadata for add_entity.
                    class_idx = entity_metadata['class']
                    class_name = entity_metadata['class_name']
                    
                    frame_info['entities'].append({
                        'id': entity_id,
                        'vector_id': qdrant_vector_id,
                        'class': class_idx,
                        'class_name': class_name,
                        # 'bbox': entity_metadata['bbox'],
                        'confidence': entity_metadata['confidence']
                    })
                # Compute relationships if at least 2 entities
                if len(frame_info['entities']) >= 2:
                    rel_entities = []
                    for ent in frame_info['entities']:
                        meta = vectorize.get_entity_metadata(ent['vector_id'])
                        if meta is not None:
                            meta['id'] = ent['id']  # Ensure logical id is present for relationships
                        rel_entities.append(meta)
                    frame_info['relationships'] = compute_relationships(rel_entities)
            dataset_frames.append(frame_info)
        if args.show is True:
            cv2.imshow('BoxMOT', img_for_display) # MODIFIED: Use the image meant for display
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
        # Store detailed speed for the current frame from results
        if hasattr(r, 'speed') and isinstance(r.speed, dict):
             # r.speed is a dictionary with keys like 'preprocess', 'inference', 'postprocess'
            frame_speeds.append(r.speed)
        else:
            # Handle cases where speed info might not be available
            frame_speeds.append({'preprocess': 0, 'inference': 0, 'postprocess': 0}) # Append zeros if speed is missing

    total_end_time = time.time() # End total timer
    total_time = total_end_time - total_start_time # This is total script execution time
    total_frames = frame_idx + 1 # Total number of frames processed

    # Calculate statistics based on 'inference' times from frame_speeds
    inference_times_ms = [s.get('inference', 0) for s in frame_speeds]
    average_frame_inference_time_ms = np.mean(inference_times_ms) if inference_times_ms else 0
    min_frame_inference_time_ms = np.min(inference_times_ms) if inference_times_ms else 0
    max_frame_inference_time_ms = np.max(inference_times_ms) if inference_times_ms else 0
    median_frame_inference_time_ms = np.median(inference_times_ms) if inference_times_ms else 0
    std_frame_inference_time_ms = np.std(inference_times_ms) if inference_times_ms else 0

    # Save dataset if requested
    if args.save_dataset:
        dataset_dir = exp_dir / 'dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / 'dataset.json'
        with open(dataset_path, 'w') as f:
            json.dump(dataset_frames, f, indent=2)

    # Save metrics if requested
    if args.metrics:
        # Convert PosixPath objects in run_parameters to strings for JSON serialization
        run_params_serializable = {}
        for key, value in vars(args).items():
            if isinstance(value, Path):
                run_params_serializable[key] = str(value)
            else:
                run_params_serializable[key] = value

        metrics_data = {
            'run_parameters': run_params_serializable, # Include all run arguments as strings
            'total_frames_processed': total_frames,
            'total_script_execution_time_seconds': round(total_time, 3), # Keep total script time in seconds
            'average_frame_inference_time_ms': round(average_frame_inference_time_ms, 3),
            'embedding_metrics': {
                'total_unique_frames_embedded': num_unique_frames_embedded,
                'total_frames_skipped_due_to_similarity': num_frames_skipped_due_to_similarity,
                'total_entity_detections_processed': num_entity_detections_processed,
                'total_entity_vectors_upserted': num_entity_vectors_upserted
            },
            'frame_inference_time_statistics_ms': {
                'min': round(min_frame_inference_time_ms, 3),
                'max': round(max_frame_inference_time_ms, 3),
                'median': round(median_frame_inference_time_ms, 3),
                'std_dev': round(std_frame_inference_time_ms, 3)
            },
            'per_frame_speed_ms': frame_speeds # Include detailed speeds for each frame
        }
        metrics_dir = exp_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
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
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--save-dataset', action='store_true',
                        help='save per-frame dataset as a JSON file under runs/track/exp/dataset/dataset.json')
    parser.add_argument('--metrics', action='store_true',
                        help='calculate and save performance metrics')
    parser.add_argument('--frame-similarity-threshold', type=float, default=0.8,
                        help='Threshold for cosine similarity between frame embeddings to consider them similar. Default is 0.98.')
    parser.add_argument('--clear-prev-runs', action='store_true',
                        help='Clear previous run data from the experiment directory (project/name) before starting.')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
