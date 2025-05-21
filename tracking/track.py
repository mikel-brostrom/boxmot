# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from pathlib import Path
import time
import json
import logging
import shutil
import os

# Consolidate os.environ settings if any are still needed globally for track.py
# os.environ["TOKENIZERS_PARALLELISM"] = "false" # This is in video_processing & vectorize, ensure consistency

from boxmot.utils import ROOT, WEIGHTS # Keep for default paths in arg parser
from tracking.video_processing import process_video # Import the new video processing function
from tracking.audio_processing import process_audio # Import the audio processing function
from tracking.link_data import link_multimodal_data # Import the cross-modal linking function
# from tracking.graphify import graphify_data # Placeholder for graph generation
from tracking import vectorize # For global Qdrant reinitialization if decided
from tracking.graphify import run_graphification # Import the graphification runner

# Configure basic logging for the main orchestrator script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Suppress verbose INFO logs from httpx client (used by Qdrant client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def run(args):
    logger.info(f"Main orchestrator (track.py) started with args: {args}")
    total_orchestration_start_time = time.time()

    # --- Global Setup Phase (e.g., Clearing previous runs, Qdrant full reinitialization) ---
    if args.clear_prev_runs:
        exp_dir_to_clear = Path(args.project) / args.name
        if exp_dir_to_clear.exists() and exp_dir_to_clear.is_dir():
            logger.info(f"Orchestrator: --clear-prev-runs flag set. Deleting directory: {exp_dir_to_clear}")
            try:
                shutil.rmtree(exp_dir_to_clear)
                logger.info(f"Orchestrator: Successfully deleted {exp_dir_to_clear}.")
            except Exception as e:
                logger.error(f"Orchestrator: Error deleting {exp_dir_to_clear}: {e}.")
        else:
            logger.info(f"Orchestrator: --clear-prev-runs flag set, but directory {exp_dir_to_clear} does not exist. Nothing to delete.")

    # Global Qdrant reinitialization (if this is the desired strategy)
    # This ensures all collections are fresh based on vectorize.py's current settings.
    # video_processing.py also calls this; consider if it should only be called once globally here.
    # For now, let video_processing.py handle its specific needs, and track.py ensures overall state if necessary.
    # If vectorize.reinitialize_collections is smart enough, multiple calls might be okay (idempotent).
    # Let's assume for now that vectorize.py handles this robustly.
    try:
        logger.info("Orchestrator: Attempting global Qdrant collection reinitialization.")
        vectorize.reinitialize_collections() # Ensure this reinitializes for both visual and upcoming audio
        logger.info("Orchestrator: Global Qdrant collections reinitialized successfully.")
    except Exception as e:
        logger.error(f"Orchestrator: Failed to reinitialize Qdrant collections globally: {e}. Terminating.")
        return

    # --- Step 1: Video Processing ---
    logger.info("Orchestrator: Starting video processing step.")
    video_processing_start_time = time.time()
    # Call the refactored video processing function
    # It will use args passed from this script's parser
    video_dataset_frames, video_metrics, video_output_paths = process_video(args)
    video_processing_time = time.time() - video_processing_start_time
    logger.info(f"Orchestrator: Video processing step completed in {video_processing_time:.2f} seconds.")
    if not video_dataset_frames:
        logger.error("Orchestrator: Video processing returned no data. Aborting further steps.")
        return
    logger.info(f"Orchestrator: Video processing generated {len(video_dataset_frames)} frame data entries.")
    logger.info(f"Orchestrator: Video output paths: {video_output_paths}")
    logger.debug(f"Orchestrator: Video metrics: {json.dumps(video_metrics, indent=2)}")

    # The `initial_dataset.json` is now expected to be at video_output_paths['initial_dataset_json']
    initial_dataset_path = video_output_paths.get('initial_dataset_json')
    if not initial_dataset_path or not Path(initial_dataset_path).exists():
        logger.error(f"Orchestrator: initial_dataset.json not found at expected path: {initial_dataset_path}. Aborting.")
        return

    # --- Step 2: Audio Processing ---
    logger.info("Orchestrator: Starting audio processing step.")
    audio_processing_start_time = time.time()
    audio_segments_data, audio_processing_metrics, audio_output_paths = process_audio(args)
    audio_processing_time = time.time() - audio_processing_start_time
    logger.info(f"Orchestrator: Audio processing step completed in {audio_processing_time:.2f} seconds.")
    logger.info(f"Orchestrator: Audio processing generated {len(audio_segments_data)} audio segments.")
    logger.info(f"Orchestrator: Audio output paths: {audio_output_paths}")
    logger.debug(f"Orchestrator: Audio metrics: {json.dumps(audio_processing_metrics, indent=2)}")

    # --- Step 3: Cross-Modal Linking ---
    logger.info("Orchestrator: Starting cross-modal linking step.")
    linking_start_time = time.time()
    
    initial_dataset_json_path = video_output_paths.get('initial_dataset_json')
    audio_segments_json_path = audio_output_paths.get('audio_segments_json')

    if not initial_dataset_json_path or not Path(initial_dataset_json_path).exists():
        logger.error(f"Orchestrator: initial_dataset.json not found at {initial_dataset_json_path} for linking. Aborting.")
        return
    if not audio_segments_json_path or not Path(audio_segments_json_path).exists():
        logger.error(f"Orchestrator: Audio segments JSON ({audio_segments_json_path}) not found for linking. Aborting.")
        return

    try:
        qdrant_client = vectorize.get_qdrant_client() # Get Qdrant client instance
        enriched_dataset_path, linking_metrics = link_multimodal_data(
            video_dataset_path=Path(initial_dataset_json_path), # Ensure it's a Path object
            audio_segments_path=Path(audio_segments_json_path), # Ensure it's a Path object
            qdrant_client=qdrant_client,
            args=args # Pass the main args object
        )
        linking_time = time.time() - linking_start_time
        if enriched_dataset_path:
            logger.info(f"Orchestrator: Cross-modal linking step completed in {linking_time:.2f} seconds.")
            logger.info(f"Orchestrator: Enriched dataset saved to: {enriched_dataset_path}")
            logger.debug(f"Orchestrator: Linking metrics: {json.dumps(linking_metrics, indent=2)}")
        else:
            logger.error(f"Orchestrator: Cross-modal linking failed. Metrics: {json.dumps(linking_metrics, indent=2)}")
            # Fallback to initial_dataset_path to allow subsequent steps to run if desired, but log severe warning.
            logger.warning("Orchestrator: Proceeding with initial_dataset_path due to linking failure. Graph data will not be enriched.")
            enriched_dataset_path = initial_dataset_path # Fallback
            # Ensure linking_metrics is what's expected by the final summary if it failed.
            # The linking_metrics from the function should already reflect the failure.

    except Exception as e:
        logger.error(f"Orchestrator: Error during cross-modal linking: {e}", exc_info=True)
        # Decide if to proceed or terminate. For now, let's try to proceed if enriched_dataset_path is somehow set or fallback.
        # If linking is critical, this should be a return.
        # Fallback to initial_dataset_path to allow subsequent steps to run if desired, but log severe warning.
        logger.warning("Orchestrator: Proceeding with initial_dataset_path due to linking error. Graph data will not be enriched.")
        enriched_dataset_path = initial_dataset_path 
        linking_metrics = {"error": str(e), "status": "failed"}

    # --- Step 4: Graphify Data ---
    logger.info("Orchestrator: Starting data graphifying step.")
    graphify_start_time = time.time()
    graph_metrics = {}
    if enriched_dataset_path: # Only proceed if we have a dataset (either enriched or fallback)
        try:
            # Ensure the args object has the path to the dataset to be graphified
            args.enriched_dataset_path = str(enriched_dataset_path) 
            graph_metrics = run_graphification(args)
            graphify_time = time.time() - graphify_start_time
            if graph_metrics.get("status") == "error":
                logger.error(f"Orchestrator: Data graphifying step failed in {graphify_time:.2f} seconds. Error: {graph_metrics.get('message')}")
            else:
                logger.info(f"Orchestrator: Data graphifying step completed in {graphify_time:.2f} seconds.")
                logger.debug(f"Orchestrator: Graphification metrics: {json.dumps(graph_metrics, indent=2)}")
        except Exception as e:
            graphify_time = time.time() - graphify_start_time
            logger.error(f"Orchestrator: Error during data graphifying: {e} after {graphify_time:.2f}s", exc_info=True)
            graph_metrics = {"status": "error", "message": str(e)}
    else:
        logger.warning("Orchestrator: Skipping graphification because enriched_dataset_path is not set.")
        graph_metrics = {"status": "skipped", "message": "Enriched dataset path not available"}

    # --- Orchestration Summary ---
    total_orchestration_time = time.time() - total_orchestration_start_time
    logger.info(f"Orchestrator: Full orchestration completed in {total_orchestration_time:.2f} seconds.")
    
    # Aggregate and save final metrics if needed
    final_metrics = {
        "orchestration_total_time_seconds": round(total_orchestration_time, 3),
        "video_processing_metrics": video_metrics,
        "audio_processing_metrics": audio_processing_metrics,
        "linking_metrics": linking_metrics,
        "graph_metrics": graph_metrics
    }
    
    # Save aggregated metrics (ensure exp_dir is robustly defined or passed)
    exp_dir = Path(args.project) / args.name
    final_metrics_dir = exp_dir / 'metrics'
    final_metrics_dir.mkdir(parents=True, exist_ok=True)
    final_metrics_path = final_metrics_dir / 'orchestration_metrics.json'
    with open(final_metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Orchestrator: Aggregated metrics saved to {final_metrics_path}")


def parse_opt():
    parser = argparse.ArgumentParser(description="Main Orchestrator for Multimodal Video Analysis")
    
    # Arguments that are primarily for video_processing.py but are passed via this orchestrator
    # These should mirror what `parse_video_proc_args` in video_processing.py expects,
    # or be a superset if track.py needs them for other purposes too.
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n.pt', help='YOLO model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='ReID model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='Tracking method (e.g., deepocsort)')
    parser.add_argument('--source', type=str, default='0', help='Video source (file, URL, webcam ID)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None, help='Inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--device', default='', help='Computation device (cuda, cpu)')
    parser.add_argument('--show', action='store_true', help='Show video processing output (passed to video_processing)')
    parser.add_argument('--save', action='store_true', help='Save tracked video output (by Ultralytics, via video_processing)')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', type=Path, help='Base directory for saving results')
    parser.add_argument('--name', default='exp', help='Experiment name, results saved to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Overwrite existing project/name if it exists')
    parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='Video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false', default=True, help='Show labels on bounding boxes')
    parser.add_argument('--show-conf', action='store_false', default=True, help='Show confidence scores on bounding boxes')
    parser.add_argument('--show-trajectories', action='store_true', default=False, help='Show tracking trajectories')
    parser.add_argument('--save-txt', action='store_true', help='Save tracking results in TXT format')
    parser.add_argument('--save-id-crops', action='store_true', help='Save cropped images of tracked IDs')
    parser.add_argument('--line-width', default=None, type=int, help='Bounding box line width')
    parser.add_argument('--per-class', default=False, action='store_true', help='Track objects per class independently')
    parser.add_argument('--verbose', default=True, action='store_false', help='Enable verbose output from video processing')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='Class-agnostic NMS')
    
    # Arguments specific to data generation and processing stages, passed to relevant modules
    parser.add_argument('--save-dataset', action='store_true', help='Enable saving of intermediate datasets (initial_dataset.json from video)')
    parser.add_argument('--metrics', action='store_true', help='Enable calculation and saving of performance metrics for each stage')
    parser.add_argument('--frame-similarity-threshold', type=float, default=0.98, help='Similarity threshold for reusing frame embeddings')
    parser.add_argument('--entity-similarity-threshold', type=float, default=0.90, help='Similarity threshold for linking entity detections across frames/ReID')
    
    # Orchestrator-specific arguments
    parser.add_argument('--clear-prev-runs', action='store_true', help='Clear previous run data from the experiment directory (project/name) before starting.')
    
    # Audio processing arguments
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN'), help='Hugging Face authentication token for audio models (e.g., pyannote). Can also be set via HF_TOKEN env var.')

    # Add placeholders for future arguments related to audio, linking, graphify if they differ from video args
    # parser.add_argument('--audio-source', type=str, default=None, help='Audio source file (if different from video or for audio-only processing)')
    # parser.add_argument('--audio-output-dir', type=Path, help='Directory to save audio processing results') # Example

    # Neo4j connection arguments
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (default: bolt://localhost:7687 or NEO4J_URI env var)."
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: neo4j or NEO4J_USER env var)."
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password (default: password or NEO4J_PASSWORD env var)."
    )

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = parse_opt()
    run(args)
