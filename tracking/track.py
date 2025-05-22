# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from pathlib import Path
import time
import json
import shutil
import os
import structlog # Import structlog

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Consolidate os.environ settings if any are still needed globally for track.py
# os.environ["TOKENIZERS_PARALLELISM"] = "false" # This is in video_processing & vectorize, ensure consistency

from boxmot.utils import ROOT, WEIGHTS # Keep for default paths in arg parser
from tracking.video_processing import process_video # Import the new video processing function
from tracking.audio_processing import process_audio # Import the audio processing function
from tracking.link_data import link_multimodal_data # Import the cross-modal linking function
# from tracking.graphify import graphify_data # Placeholder for graph generation
from tracking import vectorize # For global Qdrant reinitialization if decided
from tracking.graphify import run_graphification # Import the graphification runner

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.dev.ConsoleRenderer(), # For development, prints nicely to console
        # structlog.processors.JSONRenderer() # For production, uncomment and comment out ConsoleRenderer
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
# Suppress verbose INFO logs from httpx client (used by Qdrant client)
# This still uses standard logging, which structlog can route if needed,
# but often it's easier to configure such libraries directly.
import logging as std_logging # Keep for configuring third-party libraries
std_logging.getLogger("httpx").setLevel(std_logging.WARNING)

logger = structlog.get_logger(__name__) # Use structlog's get_logger

def run(args):
    # Set environment variable based on flag
    if args.use_captioning:
        os.environ["USE_CAPTIONING"] = "True"
        logger.info("Setting USE_CAPTIONING=True based on --use-captioning flag")
    else:
        os.environ["USE_CAPTIONING"] = os.environ.get("USE_CAPTIONING", "False")
        logger.info(f"USE_CAPTIONING environment variable: {os.environ.get('USE_CAPTIONING', 'False')}")
    
    logger.info("Main orchestrator (track.py) started.", args=vars(args)) # Log args as a dictionary
    total_orchestration_start_time = time.time()

    # --- Global Setup Phase (e.g., Clearing previous runs, Qdrant full reinitialization) ---
    if args.clear_prev_runs:
        exp_dir_to_clear = Path(args.project) / args.name
        if exp_dir_to_clear.exists() and exp_dir_to_clear.is_dir():
            logger.info("Orchestrator: --clear-prev-runs flag set. Deleting directory.", directory=str(exp_dir_to_clear))
            try:
                shutil.rmtree(exp_dir_to_clear)
                logger.info("Orchestrator: Successfully deleted directory.", directory=str(exp_dir_to_clear))
            except Exception as e:
                logger.error("Orchestrator: Error deleting directory.", directory=str(exp_dir_to_clear), error=e, exc_info=True)
        else:
            logger.info("Orchestrator: --clear-prev-runs flag set, but directory does not exist. Nothing to delete.", directory=str(exp_dir_to_clear))

    try:
        logger.info("Orchestrator: Attempting global Qdrant collection reinitialization.")
        vectorize.reinitialize_collections()
        logger.info("Orchestrator: Global Qdrant collections reinitialized successfully.")
    except Exception as e:
        logger.error("Orchestrator: Failed to reinitialize Qdrant collections globally. Terminating.", error=e, exc_info=True)
        return

    # --- Step 1: Video Processing ---
    logger.info("Orchestrator: Starting video processing step.")
    video_processing_start_time = time.time()
    video_dataset_frames, video_metrics, video_output_paths = process_video(args)
    video_processing_time = time.time() - video_processing_start_time
    logger.info("Orchestrator: Video processing step completed.", duration_seconds=round(video_processing_time, 2))
    if not video_dataset_frames:
        logger.error("Orchestrator: Video processing returned no data. Aborting further steps.")
        return
    logger.info("Orchestrator: Video processing generated frame data.", num_frames=len(video_dataset_frames))
    logger.info("Orchestrator: Video output paths.", paths=video_output_paths)
    logger.debug("Orchestrator: Video metrics.", metrics=video_metrics) # structlog handles dicts well

    initial_dataset_path = video_output_paths.get('initial_dataset_json')
    if not initial_dataset_path or not Path(initial_dataset_path).exists():
        logger.error("Orchestrator: initial_dataset.json not found. Aborting.", expected_path=initial_dataset_path)
        return

    # --- Step 2: Audio Processing ---
    logger.info("Orchestrator: Starting audio processing step.")
    audio_processing_start_time = time.time()
    audio_segments_data, audio_processing_metrics, audio_output_paths = process_audio(args)
    audio_processing_time = time.time() - audio_processing_start_time
    logger.info("Orchestrator: Audio processing step completed.", duration_seconds=round(audio_processing_time, 2))
    logger.info("Orchestrator: Audio processing generated segments.", num_segments=len(audio_segments_data))
    logger.info("Orchestrator: Audio output paths.", paths=audio_output_paths)
    logger.debug("Orchestrator: Audio metrics.", metrics=audio_processing_metrics)

    # --- Step 3: Cross-Modal Linking ---
    logger.info("Orchestrator: Starting cross-modal linking step.")
    linking_start_time = time.time()
    linking_time = None  # Initialize linking_time
    linking_metrics = {"status": "pending", "message": "Linking not initiated"} # Initialize linking_metrics
    
    initial_dataset_json_path = video_output_paths.get('initial_dataset_json')
    audio_segments_json_path = audio_output_paths.get('audio_segments_json')

    if not initial_dataset_json_path or not Path(initial_dataset_json_path).exists():
        logger.error("Orchestrator: initial_dataset.json not found for linking. Aborting.", path=initial_dataset_json_path)
        return
    if not audio_segments_json_path or not Path(audio_segments_json_path).exists():
        logger.error("Orchestrator: Audio segments JSON not found for linking. Aborting.", path=audio_segments_json_path)
        return

    enriched_dataset_path = None # Initialize to ensure it's defined
    linking_metrics = {} # Initialize to ensure it's defined
    try:
        qdrant_client = vectorize.get_qdrant_client() # Get Qdrant client instance
        enriched_dataset_path, linking_metrics = link_multimodal_data(
            video_dataset_path=Path(initial_dataset_json_path), # Ensure it's a Path object
            audio_segments_path=Path(audio_segments_json_path), # Ensure it's a Path object
            qdrant_client=qdrant_client,
            args=args # Pass the main args object
        )
        linking_time = time.time() - linking_start_time # Calculate time after the operation
        if enriched_dataset_path:
            logger.info(f"Orchestrator: Cross-modal linking step completed in {linking_time:.2f} seconds.")
            logger.info(f"Orchestrator: Enriched dataset saved to: {enriched_dataset_path}")
        else:
            # The linking_metrics from the function should already reflect the failure.
            logger.error(f"Orchestrator: Cross-modal linking failed in {linking_time:.2f} seconds. Metrics: {linking_metrics}")
            logger.warning("Orchestrator: Proceeding with initial_dataset_path due to linking failure. Graph data will not be enriched.")
            enriched_dataset_path = initial_dataset_json_path # Fallback
            # linking_metrics should already be set by the failing function

    except Exception as e:
        linking_time = time.time() - linking_start_time # Calculate time until exception
        logger.error(f"Orchestrator: Error during cross-modal linking: {e}", exc_info=True)
        # Decide if to proceed or terminate. For now, let's try to proceed if enriched_dataset_path is somehow set or fallback.
        # If linking is critical, this should be a return.
        # Fallback to initial_dataset_path to allow subsequent steps to run if desired, but log severe warning.
        logger.warning("Orchestrator: Proceeding with initial_dataset_path due to linking error. Graph data will not be enriched.")
        enriched_dataset_path = initial_dataset_path 
        linking_metrics = {"error": str(e), "status": "failed_exception"}

    # --- Step 4: Graphify Data ---
    logger.info("Orchestrator: Starting data graphifying step.")
    graphify_start_time = time.time()
    graph_metrics = {}
    graphify_time = None # Initialize graphify_time

    if enriched_dataset_path: # Only proceed if we have a dataset (either enriched or fallback)
        try:
            # Ensure the args object has the path to the dataset to be graphified
            args.enriched_dataset_path = str(enriched_dataset_path) 
            graph_metrics = run_graphification(args)
            graphify_time = time.time() - graphify_start_time # Calculate time after the operation
            if graph_metrics.get("status") == "error":
                logger.error(f"Orchestrator: Data graphifying step failed in {graphify_time:.2f} seconds. Error: {graph_metrics.get('message')}")
            else:
                logger.info(f"Orchestrator: Data graphifying step completed in {graphify_time:.2f} seconds.")
                logger.debug(f"Orchestrator: Graphification metrics: {json.dumps(graph_metrics, indent=2)}")
        except Exception as e:
            graphify_time = time.time() - graphify_start_time # Calculate time until exception
            logger.error(f"Orchestrator: Error during data graphifying: {e} after {graphify_time:.2f}s", exc_info=True)
            graph_metrics = {"status": "error_exception", "message": str(e)}
    else:
        logger.warning("Orchestrator: Skipping graphification because enriched_dataset_path is not set.")
        graph_metrics = {"status": "skipped", "message": "Enriched dataset path not available"}
        graphify_time = time.time() - graphify_start_time # Still record time taken to decide to skip

    # --- Orchestration Summary ---
    total_orchestration_time = time.time() - total_orchestration_start_time
    logger.info("Orchestrator: Full orchestration completed.", total_duration_seconds=round(total_orchestration_time, 2))
    
    # Aggregate and save final metrics
    # We might change this later to log metrics as events throughout the process
    # and then perhaps have a separate script to aggregate them if needed,
    # or use a metrics backend that handles aggregation.
    # For now, keep the existing final JSON dump.
    final_metrics = {
        "orchestration_total_time_seconds": round(total_orchestration_time, 3),
        "video_processing_metrics": video_metrics, # These should be dicts already
        "audio_processing_metrics": audio_processing_metrics, # These should be dicts already
        "linking_metrics": linking_metrics, # This should be a dict
        "graph_metrics": graph_metrics # This should be a dict
    }
    
    exp_dir = Path(args.project) / args.name
    final_metrics_dir = exp_dir / 'metrics'
    final_metrics_dir.mkdir(parents=True, exist_ok=True)
    final_metrics_path = final_metrics_dir / 'orchestration_metrics.json'
    try:
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Orchestrator: Aggregated metrics saved to {final_metrics_path}")
    except Exception as e:
        logger.error("Orchestrator: Failed to save aggregated metrics.", path=str(final_metrics_path), error=e, exc_info=True)

    # --- Orchestration Timings Summary ---
    print("\n" + "="*50)
    print("   ORCHESTRATION TIMINGS SUMMARY")
    print("="*50)
    print(f"Video Processing Time: {video_processing_time:.2f} seconds")
    print(f"Audio Processing Time: {audio_processing_time:.2f} seconds")
    
    if linking_time is not None:
        status_msg = f"(Status: {linking_metrics.get('status', 'unknown')})"
        print(f"Cross-Modal Linking Time: {linking_time:.2f} seconds {status_msg}")
    else:
        print(f"Cross-Modal Linking: Not performed or failed before timing (Status: {linking_metrics.get('status', 'unknown')}).")

    if graphify_time is not None:
        status_msg = f"(Status: {graph_metrics.get('status', 'unknown')})"
        print(f"Data Graphification Time: {graphify_time:.2f} seconds {status_msg}")
    elif graph_metrics.get("status") == "skipped":
        print("Data Graphification: Skipped (input data not available)")
    else:
        print(f"Data Graphification: Not performed or failed before timing (Status: {graph_metrics.get('status', 'unknown')}).")
        
    print(f"Total Orchestration Time: {total_orchestration_time:.2f} seconds")
    print("="*50)
    
    # Also log to structured log
    logger.info("--- Orchestration Timings Summary ---")
    logger.info(f"Video Processing Time: {video_processing_time:.2f} seconds")
    logger.info(f"Audio Processing Time: {audio_processing_time:.2f} seconds")
    
    if linking_time is not None:
        status_msg = f"(Status: {linking_metrics.get('status', 'unknown')})"
        logger.info(f"Cross-Modal Linking Time: {linking_time:.2f} seconds {status_msg}")
    else:
        logger.info(f"Cross-Modal Linking: Not performed or failed before timing (Status: {linking_metrics.get('status', 'unknown')}).")

    if graphify_time is not None:
        status_msg = f"(Status: {graph_metrics.get('status', 'unknown')})"
        logger.info(f"Data Graphification Time: {graphify_time:.2f} seconds {status_msg}")
    elif graph_metrics.get("status") == "skipped":
        logger.info("Data Graphification: Skipped (input data not available)")
    else:
        logger.info(f"Data Graphification: Not performed or failed before timing (Status: {graph_metrics.get('status', 'unknown')}).")
        
    logger.info(f"Total Orchestration Time: {total_orchestration_time:.2f} seconds")
    logger.info("-----------------------------------")


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
    parser.add_argument('--use-captioning', action='store_true', help='Use image captioning instead of storing images in MinIO. Sets USE_CAPTIONING=True environment variable.')
    parser.add_argument(
        '--frame-similarity-threshold', 
        type=float, 
        default=float(os.getenv('FRAME_SIMILARITY_THRESHOLD', 0.8)), 
        help='Similarity threshold for reusing frame embeddings. Env: FRAME_SIMILARITY_THRESHOLD (default: 0.98)'
    )
    parser.add_argument(
        '--entity-similarity-threshold', 
        type=float, 
        default=float(os.getenv('ENTITY_SIMILARITY_THRESHOLD', 0.8)), 
        help='Similarity threshold for linking entity detections across frames/ReID. Env: ENTITY_SIMILARITY_THRESHOLD (default: 0.95)'
    )
    
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
