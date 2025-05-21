"""
Handles the core video processing tasks including object detection, tracking, 
ReID, visual embedding generation, and saving video-related outputs.
"""

import argparse
from functools import partial
from pathlib import Path
import time
import json
import numpy as np
import logging
import shutil
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS # Assuming these are still relevant or will be adjusted
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model)
from tracking.relate import compute_relationships
from tracking import vectorize # This will be used for visual embeddings

checker = RequirementsChecker()
# Ensure these are relevant for video processing standalone or adjust if they become part of a larger setup
checker.check_packages((
    'ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',
    'qdrant-client',
    'transformers',
    'torch',
    'Pillow'
))

from ultralytics import YOLO

# Configure basic logging - might be configured by the calling script (track.py)
# For now, set a basic config if this module is run directly or for testing.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        if hasattr(tracker, 'model'): # motion only models do not have
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def process_video(args):
    """
    Processes the input video source for object detection, tracking, and ReID.
    Generates visual embeddings for frames and entities.
    Calculates spatial relationships between entities.
    Saves tracked video, entity crops, and a dataset JSON file.
    Returns:
        list: A list of dictionaries, where each dictionary represents a processed frame's data.
        dict: A dictionary containing performance metrics for the video processing.
        dict: A dictionary containing paths to the saved output files (e.g., video, dataset JSON).
    """
    logging.info(f"Starting video processing with args: {args}")

    # Clear previous run data if flag is set (This might be managed by the main orchestrator)
    # For now, keeping it here if process_video is intended to be somewhat self-contained for an experiment
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

    # Reinitialize Qdrant collections (visual part) at the start of the run
    # This assumes vectorize.py has been updated to handle targeted reinitialization
    # or that track.py will manage overall reinitialization.
    # For now, let's assume visual collections are reinitialized here if needed.
    try:
        # Consider if reinitialize_collections should be more granular, e.g., reinitialize_visual_collections()
        vectorize.reinitialize_collections() # This might reinit all, including audio if not careful
        logging.info("Successfully reinitialized Qdrant collections for video processing.")
    except Exception as e:
        logging.error(f"Failed to reinitialize Qdrant collections for video: {e}. Terminating video processing.")
        return [], {}, {}


    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt', # Default fallback if not an ultralytics model (though this logic might need refinement)
    )

    # The results generator
    results_generator = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False, # show is handled by the main orchestrator if needed, or CLI arg for this script
        stream=True, # Essential for frame-by-frame processing
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save, # This controls saving the output video by ultralytics
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width,
        save_crop=args.save_id_crops # This controls saving individual crops by ultralytics
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        m = get_yolo_inferer(args.yolo_model)
        yolo_model_instance = m(model=args.yolo_model, device=yolo.predictor.device,
                                args=yolo.predictor.args) # yolo.predictor.args might need careful checking
        yolo.predictor.model = yolo_model_instance
        yolo.add_callback(
            "on_predict_batch_start",
            lambda p: yolo_model_instance.update_im_paths(p)
        )
        yolo.predictor.preprocess = (
            lambda imgs: yolo_model_instance.preprocess(im=imgs))
        yolo.predictor.postprocess = (
            lambda preds, im, im0s:
            yolo_model_instance.postprocess(preds=preds, im=im, im0s=im0s))

    yolo.predictor.custom_args = args # Store args for access in callbacks like on_predict_start

    video_dataset_frames = []
    # Determine the experiment directory used by YOLO for saving outputs
    # This logic might need to be robust if yolo.predictor.save_dir is not immediately set
    # or if run is called multiple times. For a single call, it should be fine.
    # We expect yolo.track to set this up.
    
    # Fallback if save_dir is not set by predictor immediately (should be, but as a precaution)
    exp_dir = Path(args.project) / args.name 
    # After yolo.track() starts or if yolo.predictor.setup_dirs() was called, save_dir is set.
    # It's safer to get it after the first iteration or ensure yolo.predictor.save_dir is populated.
    # For now, we assume yolo.track() call above populates yolo.predictor.save_dir correctly.
    # If not, exp_dir will default to args.project/args.name.
    # It's crucial that yolo.predictor.save_dir is correctly obtained.
    # Let's defer definitive assignment of exp_dir until after the first frame if possible,
    # or rely on yolo.predictor.save_dir being set by the yolo.track() call.

    total_start_time = time.time()
    frame_speeds = []

    last_stored_frame_embedding = None
    last_stored_frame_vector_id = None
    previous_entity_count = -1
    is_first_frame = True
    
    num_unique_frames_embedded = 0
    num_frames_skipped_due_to_similarity = 0
    num_entity_detections_processed = 0
    num_entity_vectors_upserted = 0 # Qdrant upserts for entities

    # The main loop:
    for frame_idx, r in enumerate(results_generator):
        if frame_idx == 0: # After the first result, save_dir should be set
            exp_dir = Path(yolo.predictor.save_dir)
            logging.info(f"Experiment directory for outputs: {exp_dir}")

        clean_frame = r.orig_img.copy()
        # annotated_frame_for_video is modified by plot_results
        annotated_frame_for_video = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
        
        frame_vector_id_for_json = None

        if args.save_dataset: # This flag now controls the creation of the (initial) dataset
            try:
                current_frame_actual_embedding = vectorize.embed_image(clean_frame)
                current_entity_count = len(r.boxes) if r.boxes is not None and hasattr(r.boxes, 'id') and r.boxes.id is not None else 0
                should_store_new_frame_embedding = True

                if not is_first_frame and last_stored_frame_embedding is not None:
                    similarity = vectorize.calculate_cosine_similarity(
                        current_frame_actual_embedding,
                        last_stored_frame_embedding
                    )
                    if similarity >= args.frame_similarity_threshold and current_entity_count == previous_entity_count:
                        frame_vector_id_for_json = last_stored_frame_vector_id
                        should_store_new_frame_embedding = False
                        logging.debug(f"Frame {frame_idx}: Reusing frame vector ID {frame_vector_id_for_json}. Sim: {similarity:.4f}, Entities: {current_entity_count}")
                        if not is_first_frame:
                             num_frames_skipped_due_to_similarity += 1
                
                if should_store_new_frame_embedding:
                    new_frame_vector_id = vectorize.add_frame_embedding(
                        embedding=current_frame_actual_embedding,
                        frame_idx=frame_idx,
                        timestamp=frame_idx, # Placeholder timestamp
                        image=clean_frame # Store image if configured in vectorize.py
                    )
                    frame_vector_id_for_json = new_frame_vector_id
                    last_stored_frame_embedding = current_frame_actual_embedding
                    last_stored_frame_vector_id = new_frame_vector_id
                    num_unique_frames_embedded +=1
                    logging.debug(f"Frame {frame_idx}: Stored new frame vector ID {frame_vector_id_for_json}. Entities: {current_entity_count}")
                
                previous_entity_count = current_entity_count
                if is_first_frame:
                    is_first_frame = False
            except Exception as e:
                logging.error(f"Error processing frame {frame_idx} for visual embedding: {e}", exc_info=True)
                frame_vector_id_for_json = None

            frame_data_for_json = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx, # Placeholder, to be refined
                'geo': None, # Placeholder
                'frame_vector_id': frame_vector_id_for_json,
                'entities': [],
                'relationships': []
            }

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().tolist()
                ids = r.boxes.id.cpu().numpy().tolist() if r.boxes.id is not None else [None] * len(boxes)
                confs = r.boxes.conf.cpu().numpy().tolist()
                clss = r.boxes.cls.cpu().numpy().tolist()
                names = r.names

                processed_entities_for_relationships = []

                for i in range(len(boxes)):
                    bbox = boxes[i]
                    crop_img = clean_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    tracker_id = ids[i]
                    
                    if tracker_id is None:
                        logging.warning(f"Skipping detection with no tracker ID at frame {frame_idx}, detection {i}")
                        continue
                    num_entity_detections_processed += 1
                    
                    # current_detection_class_idx = clss[i] if clss and i < len(clss) else None
                    # current_detection_class_name = names.get(int(clss[i])) if names and current_detection_class_idx is not None and int(clss[i]) in names else "unknown"
                    
                    entity_logical_id = str(tracker_id) # Initial logical ID is the tracker ID
                    entity_qdrant_vector_id = None
                    
                    # Embed entity crop
                    entity_embedding = vectorize.embed_crop(crop_img)

                    # Search for existing logical entity to link or update
                    found_logical_id, _, payload_from_search, _ = vectorize.search_entity(entity_embedding, threshold=args.entity_similarity_threshold) # Add entity_similarity_threshold to args

                    final_class_idx = clss[i]
                    final_class_name = names.get(int(clss[i]))

                    if found_logical_id:
                        entity_logical_id = found_logical_id # Use existing logical ID
                        if payload_from_search: # Update class info from Qdrant if more reliable
                            final_class_idx = payload_from_search.get('class', final_class_idx)
                            final_class_name = payload_from_search.get('class_name', final_class_name)
                    
                    entity_metadata_for_qdrant = {
                        'bbox': bbox, # Bbox in current frame
                        'id': entity_logical_id, # The logical ID (tracker or found) - for Qdrant payload, not Qdrant point ID
                        'class': final_class_idx,
                        'class_name': final_class_name,
                        'confidence': confs[i]
                        # Any other persistent info for this logical entity
                    }

                    # Add/Update entity in Qdrant. vectorize.add_entity should handle upsert logic.
                    # It creates a NEW Qdrant point for EACH detection's embedding, but payload contains the LOGICAL entity_id.
                    entity_qdrant_vector_id = vectorize.add_entity(
                        entity_embedding,
                        crop_img,
                        entity_metadata_for_qdrant,
                        entity_id=entity_logical_id # Pass the logical entity ID for payload
                    )
                    num_entity_vectors_upserted +=1

                    # For dataset.json and relationship calculation
                    entity_info_for_json = {
                        'id': entity_logical_id, # Logical ID
                        'vector_id': entity_qdrant_vector_id, # Qdrant point ID for this specific detection instance/embedding
                        'class': final_class_idx,
                        'class_name': final_class_name,
                        'confidence': confs[i]
                        # No bbox here, as it's frame-specific and relate.py will get it from metadata
                    }
                    frame_data_for_json['entities'].append(entity_info_for_json)
                    
                    # For relationship calculation, we need the bbox in the current frame
                    # We fetch from Qdrant to ensure consistency, or pass metadata directly.
                    # Let's use metadata directly since we just prepared it.
                    # Ensure 'bbox' is part of the metadata used by compute_relationships
                    temp_entity_for_rel = entity_metadata_for_qdrant.copy() # Includes current bbox
                    temp_entity_for_rel['id'] = entity_logical_id # Ensure logical ID is used
                    processed_entities_for_relationships.append(temp_entity_for_rel)


                if len(processed_entities_for_relationships) >= 2:
                    frame_data_for_json['relationships'] = compute_relationships(processed_entities_for_relationships)
            
            video_dataset_frames.append(frame_data_for_json)

        if args.show is True: # For standalone testing of video_processing
             cv2.imshow('Video Processing Output', annotated_frame_for_video)
             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
        
        if hasattr(r, 'speed') and isinstance(r.speed, dict):
            frame_speeds.append(r.speed)
        else:
            frame_speeds.append({'preprocess': 0, 'inference': 0, 'postprocess': 0})

    # End of main loop
    if args.show:
        cv2.destroyAllWindows()

    total_end_time = time.time()
    total_script_time_seconds = total_end_time - total_start_time
    total_frames_processed = frame_idx + 1 if frame_idx is not None else 0


    # --- Prepare Metrics ---
    inference_times_ms = [s.get('inference', 0) for s in frame_speeds]
    avg_inf_time = np.mean(inference_times_ms) if inference_times_ms else 0
    min_inf_time = np.min(inference_times_ms) if inference_times_ms else 0
    max_inf_time = np.max(inference_times_ms) if inference_times_ms else 0
    median_inf_time = np.median(inference_times_ms) if inference_times_ms else 0
    std_inf_time = np.std(inference_times_ms) if inference_times_ms else 0

    run_params_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    video_metrics = {
        'run_parameters': run_params_serializable,
        'total_frames_processed': total_frames_processed,
        'total_video_processing_time_seconds': round(total_script_time_seconds, 3),
        'average_frame_inference_time_ms': round(avg_inf_time, 3),
        'embedding_metrics': {
            'total_unique_frames_embedded': num_unique_frames_embedded,
            'total_frames_skipped_due_to_similarity': num_frames_skipped_due_to_similarity,
            'total_entity_detections_processed': num_entity_detections_processed,
            'total_entity_vectors_upserted': num_entity_vectors_upserted
        },
        'frame_inference_time_statistics_ms': {
            'min': round(min_inf_time, 3),
            'max': round(max_inf_time, 3),
            'median': round(median_inf_time, 3),
            'std_dev': round(std_inf_time, 3)
        },
        'per_frame_speed_ms': frame_speeds
    }
    
    output_paths = {}
    # --- Save Dataset (initial_dataset.json) ---
    if args.save_dataset:
        dataset_dir = exp_dir / 'dataset' # exp_dir should be correctly set now
        dataset_dir.mkdir(parents=True, exist_ok=True)
        # This will be the "initial" dataset, focused on video processing outputs
        initial_dataset_path = dataset_dir / 'initial_dataset.json' 
        with open(initial_dataset_path, 'w') as f:
            json.dump(video_dataset_frames, f, indent=2)
        logging.info(f"Initial video dataset saved to: {initial_dataset_path}")
        output_paths['initial_dataset_json'] = str(initial_dataset_path)

    # --- Save Video Metrics ---
    # This might be aggregated later by the main orchestrator
    # For now, video_processing.py can save its own metrics.
    if args.metrics: # Assuming args.metrics controls saving of these specific metrics
        metrics_dir = exp_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        video_metrics_path = metrics_dir / 'video_processing_metrics.json'
        with open(video_metrics_path, 'w') as f:
            json.dump(video_metrics, f, indent=2)
        logging.info(f"Video processing metrics saved to: {video_metrics_path}")
        output_paths['video_metrics_json'] = str(video_metrics_path)
    
    # Other output paths if yolo.track saved video or crops
    if args.save: # If ultralytics saved a video
        # Need to determine the actual video path. It's usually in exp_dir.
        # Ultralytics names it based on the source.
        # This is a guess; ultralytics might have a more direct way to get this path.
        src_path = Path(args.source)
        video_filename = src_path.stem + ".avi" # or .mp4, depends on ultralytics default
        # A more robust way would be to find the video file in exp_dir
        saved_video_files = list(exp_dir.glob(f'{src_path.stem}.*'))
        if saved_video_files:
             output_paths['tracked_video'] = str(saved_video_files[0])
        else:
             logging.warning(f"Tracked video file not found in {exp_dir} as expected.")


    if args.save_id_crops:
        output_paths['entity_crops_dir'] = str(exp_dir / 'crops') # Ultralytics saves crops here

    logging.info(f"Video processing finished. Returning {len(video_dataset_frames)} frames data.")
    return video_dataset_frames, video_metrics, output_paths


# Placeholder for argument parsing if this script is run standalone for testing
def parse_video_proc_args():
    parser = argparse.ArgumentParser(description="Video Processing Module")
    # Add arguments previously in track.py's parse_opt that are relevant to video processing
    # This is a subset for now, expand as needed.
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n.pt')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--imgsz', nargs='+', type=int, default=None)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--device', default='')
    parser.add_argument('--show', action='store_true', help="Show video output (for standalone testing)")
    parser.add_argument('--save', action='store_true', help="Save tracked video output by Ultralytics")
    parser.add_argument('--classes', nargs='+', type=int, default=None)
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', type=Path)
    parser.add_argument('--name', default='exp_video_proc', type=str) # Different default name for standalone
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)
    parser.add_argument('--show-labels', action='store_false', default=True)
    parser.add_argument('--show-conf', action='store_false', default=True)
    parser.add_argument('--show-trajectories', action='store_true', default=False)
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-id-crops', action='store_true')
    parser.add_argument('--line-width', default=None, type=int)
    parser.add_argument('--per-class', default=False, action='store_true')
    parser.add_argument('--verbose', default=True, action='store_false') # Default to True for verbose if standalone
    parser.add_argument('--agnostic-nms', default=False, action='store_true')
    
    # Args that were specific to the main script, now for video_processing
    parser.add_argument('--save-dataset', action='store_true', help='Save per-frame dataset (initial_dataset.json)')
    parser.add_argument('--metrics', action='store_true', help='Calculate and save video processing performance metrics')
    parser.add_argument('--frame-similarity-threshold', type=float, default=0.98, help='Threshold for frame similarity')
    parser.add_argument('--entity-similarity-threshold', type=float, default=0.90, help='Threshold for entity ReID/linking similarity') # New arg
    parser.add_argument('--clear-prev-runs', action='store_true', help='Clear previous run data from the experiment directory')

    return parser.parse_args()

if __name__ == '__main__':
    # This allows testing video_processing.py independently
    print("Running video_processing.py as a standalone script.")
    # Configure basic logging for standalone run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Suppress verbose INFO logs from httpx client for standalone runs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_video_proc_args()
    
    # Example: Ensure WEIGHTS and ROOT are defined if not imported correctly or paths are relative
    if not WEIGHTS.exists() :
        logging.warning(f"WEIGHTS path {WEIGHTS} might be incorrect for standalone run. Adjust if needed.")
    if not TRACKER_CONFIGS.exists():
        logging.warning(f"TRACKER_CONFIGS path {TRACKER_CONFIGS} might be incorrect. Adjust if needed.")
        
    # Manually set some args for a quick test if needed
    # args.source = "path/to/your/test/video.mp4" # Or use webcam '0'
    # args.save_dataset = True
    # args.metrics = True
    # args.save = True
    # args.show = True 
    # args.name = "video_proc_standalone_test"

    if not Path(args.source).exists() and args.source != '0':
        logging.error(f"Source file {args.source} does not exist. Exiting standalone test.")
    else:
        logging.info(f"Starting standalone video processing test with source: {args.source}")
        dataset, metrics, paths = process_video(args)
        logging.info(f"Standalone video processing completed.")
        logging.info(f"Frames processed: {len(dataset)}")
        logging.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        logging.info(f"Output paths: {paths}") 