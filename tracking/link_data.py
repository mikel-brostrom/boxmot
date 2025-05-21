"""
Cross-Modal Linker

This module is responsible for linking data from different modalities (video and audio)
based on temporal overlaps and other relevant heuristics.

It consumes:
1.  `initial_dataset.json`: Output from video processing, containing frame-by-frame 
    information including timestamps, detected entities, and their visual features.
2.  `*_audio_segments.json`: Output from audio processing, containing diarized and 
    transcribed audio segments with timestamps and speaker labels.

It produces:
1.  An `enriched_dataset.json`: This dataset augments the `initial_dataset.json` by 
    adding links to relevant audio segments for each frame (e.g., `last_audio_segment_id`).
2.  Updates to Qdrant collections:
    -   `frames` collection: Payloads are updated with `overlapping_audio_segment_ids`.
    -   `audio_transcripts` collection: Payloads are updated with `overlapping_frame_ids` 
        and `overlapping_entity_ids`.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient # For type hinting and direct use
# Import the existing client instance from vectorize.py if it's globally accessible
# from tracking.vectorize import qdrant as qdrant_global_client 

logger = logging.getLogger(__name__)


def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Loads data from a JSON file."""
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def link_multimodal_data(
    video_dataset_path: Path,
    audio_segments_path: Path,
    qdrant_client: Optional[QdrantClient] = None, # Accept Qdrant client instance
    args: Optional[Any] = None  # Add args to accept the arguments object
) -> tuple[Optional[Path], dict]:
    """
    Main function to perform cross-modal linking.
    Args:
        video_dataset_path: Path to initial_dataset.json from video processing.
        audio_segments_path: Path to *_audio_segments.json from audio processing.
        qdrant_client: Optional Qdrant client instance for updating payloads.
        args: Command line arguments (or an object mimicking it) for path configurations.

    Returns:
        tuple[Optional[Path], dict]: Path to the enriched dataset JSON and a metrics dictionary.
    """
    logger.info("Starting cross-modal linking process...")
    linking_metrics = {"status": "started", "errors": []}

    if args is None:
        logger.error("Args object not provided to link_multimodal_data. Cannot determine output path.")
        linking_metrics["status"] = "failed"
        linking_metrics["errors"].append("Args object not provided.")
        return None, linking_metrics

    # Determine output path for enriched_dataset.json using args
    base_output_dir = Path(args.project) / args.name
    dataset_output_dir = base_output_dir / "dataset"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    output_enriched_dataset_path = dataset_output_dir / "enriched_dataset.json"
    linking_metrics["enriched_dataset_output_path"] = str(output_enriched_dataset_path)

    if qdrant_client is None:
        logger.warning("Qdrant client not provided. Payload updates to Qdrant will be skipped.")

    video_data = load_json_data(video_dataset_path)
    audio_data = load_json_data(audio_segments_path)

    if video_data is None or audio_data is None:
        logger.error("Missing video or audio data. Aborting linking.")
        linking_metrics["status"] = "failed"
        linking_metrics["errors"].append("Missing video or audio input data.")
        return None, linking_metrics

    logger.info("Linking audio segments to video frames...")
    enriched_video_data = []

    for frame in video_data:
        frame_timestamp = frame.get('timestamp')
        if frame_timestamp is None:
            logger.warning(f"Frame missing timestamp: {frame.get('frame_idx', 'Unknown frame')}. Skipping.")
            enriched_video_data.append(frame.copy())
            continue

        current_frame_data = frame.copy()
        current_frame_data['overlapping_audio_segment_ids'] = []
        current_frame_data['last_audio_segment_id'] = None
        latest_end_time_for_last_segment = -1.0

        for audio_segment in audio_data:
            audio_start = audio_segment.get('start_time')
            audio_end = audio_segment.get('end_time')
            audio_id = audio_segment.get('audio_segment_unique_id')

            if audio_start is None or audio_end is None or audio_id is None:
                continue

            if audio_start <= frame_timestamp <= audio_end:
                current_frame_data['overlapping_audio_segment_ids'].append(audio_id)
            
            if audio_end <= frame_timestamp:
                if audio_end > latest_end_time_for_last_segment:
                    latest_end_time_for_last_segment = audio_end
                    current_frame_data['last_audio_segment_id'] = audio_id
                elif audio_end == latest_end_time_for_last_segment:
                    if current_frame_data['last_audio_segment_id'] and audio_id < current_frame_data['last_audio_segment_id']:
                         current_frame_data['last_audio_segment_id'] = audio_id
        
        enriched_video_data.append(current_frame_data)
        
        # Update Qdrant 'frames' collection payload
        frame_qdrant_id = current_frame_data.get('frame_vector_id') # This ID is from when frame was added to Qdrant
        if qdrant_client and frame_qdrant_id and current_frame_data['overlapping_audio_segment_ids']:
            try:
                payload_to_set = {
                    "overlapping_audio_segment_ids": current_frame_data['overlapping_audio_segment_ids']
                }
                qdrant_client.set_payload(
                    collection_name="frames", 
                    payload=payload_to_set, 
                    points=[frame_qdrant_id],
                    wait=True # Changed to True for debugging
                )
                logger.debug(f"Updated Qdrant frame {frame_qdrant_id} with overlapping audio IDs.")
            except Exception as e_q_frame:
                logger.error(f"Error updating Qdrant frame {frame_qdrant_id} payload: {e_q_frame}")

    logger.info("Finished linking audio to frames and updating Qdrant frame payloads.")

    logger.info("Linking video frames and entities to audio segments for Qdrant audio_transcripts updates...")
    for audio_segment in audio_data:
        audio_qdrant_id = audio_segment.get('transcript_vector_id') # This ID is from when audio transcript was added
        audio_start = audio_segment.get('start_time')
        audio_end = audio_segment.get('end_time')

        if audio_qdrant_id is None or audio_start is None or audio_end is None:
            continue

        overlapping_frame_indices = []
        overlapping_entity_ids_in_segment = set()

        for frame in enriched_video_data:
            frame_timestamp = frame.get('timestamp')
            frame_idx = frame.get('frame_idx')
            if frame_timestamp is None or frame_idx is None: continue

            if audio_start <= frame_timestamp <= audio_end:
                overlapping_frame_indices.append(frame_idx)
                for entity in frame.get('entities', []):
                    entity_id = entity.get('id')
                    if entity_id:
                        overlapping_entity_ids_in_segment.add(str(entity_id))
        
        if qdrant_client and (overlapping_frame_indices or overlapping_entity_ids_in_segment):
            try:
                payload_to_set = {}
                if overlapping_frame_indices:
                    payload_to_set["overlapping_frame_ids"] = sorted(list(overlapping_frame_indices))
                if overlapping_entity_ids_in_segment:
                     payload_to_set["overlapping_entity_ids"] = sorted(list(overlapping_entity_ids_in_segment))
                
                if payload_to_set: # Only update if there's something to add
                    qdrant_client.set_payload(
                        collection_name="audio_transcripts", 
                        payload=payload_to_set, 
                        points=[audio_qdrant_id],
                        wait=True # Changed to True for debugging
                    )
                    logger.debug(f"Updated Qdrant audio_transcript {audio_qdrant_id} with frame/entity IDs.")
            except Exception as e_q_audio:
                logger.error(f"Error updating Qdrant audio_transcript {audio_qdrant_id} payload: {e_q_audio}")

    logger.info("Finished Qdrant audio_transcripts payload updates.")

    logger.info(f"Saving enriched dataset to {output_enriched_dataset_path}...")
    try:
        output_enriched_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_enriched_dataset_path, 'w') as f:
            json.dump(enriched_video_data, f, indent=2)
        logger.info(f"Enriched dataset saved successfully to {output_enriched_dataset_path}")
        linking_metrics["status"] = "completed_successfully"
        return output_enriched_dataset_path, linking_metrics
    except Exception as e:
        logger.error(f"Error saving enriched_dataset.json to {output_enriched_dataset_path}: {e}")
        linking_metrics["status"] = "failed"
        linking_metrics["errors"].append(f"Error saving enriched_dataset.json: {e}")
        return None, linking_metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running link_data.py standalone test.")

    project_dir = Path("runs/track/linker_test_exp")
    video_data_dir = project_dir / "dataset"
    audio_data_dir = project_dir / "audio"
    video_data_dir.mkdir(parents=True, exist_ok=True)
    audio_data_dir.mkdir(parents=True, exist_ok=True)

    dummy_video_dataset_path = video_data_dir / "initial_dataset.json"
    dummy_audio_segments_path = audio_data_dir / "testvideo_audio_segments.json"

    sample_video_data = [
        {"frame_idx": 0, "timestamp": 0.0, "entities": [{"id": "p1", "class": "person"}], "frame_vector_id": "fv0"},
        {"frame_idx": 1, "timestamp": 0.5, "entities": [{"id": "p1", "class": "person"}, {"id": "c1", "class": "car"}], "frame_vector_id": "fv1"},
        {"frame_idx": 2, "timestamp": 1.0, "entities": [{"id": "c1", "class": "car"}], "frame_vector_id": "fv2"},
        {"frame_idx": 3, "timestamp": 1.5, "entities": [], "frame_vector_id": "fv3"},
        {"frame_idx": 4, "timestamp": 2.0, "entities": [{"id": "p2", "class": "person"}], "frame_vector_id": "fv4"}
    ]
    with open(dummy_video_dataset_path, 'w') as f: json.dump(sample_video_data, f, indent=2)

    sample_audio_data = [
        {"audio_segment_unique_id": "as1", "speaker_label": "S0", "start_time": 0.2, "end_time": 0.8, "transcript_text": "Hello there", "transcript_vector_id": "asv1"},
        {"audio_segment_unique_id": "as2", "speaker_label": "S1", "start_time": 0.9, "end_time": 1.6, "transcript_text": "General Kenobi", "transcript_vector_id": "asv2"},
        {"audio_segment_unique_id": "as3", "speaker_label": "S0", "start_time": 1.8, "end_time": 2.5, "transcript_text": "You are a bold one", "transcript_vector_id": "asv3"}
    ]
    with open(dummy_audio_segments_path, 'w') as f: json.dump(sample_audio_data, f, indent=2)

    logger.info(f"Attempting to link data: Video: {dummy_video_dataset_path}, Audio: {dummy_audio_segments_path}")
    
    # For standalone test, we'd need to initialize a Qdrant client here if we want to test Qdrant updates.
    # For now, Qdrant updates will only run if a client is passed, which track.py will do.
    # from tracking.vectorize import qdrant as test_qdrant_client # Example: if qdrant from vectorize is usable
    test_qdrant_client = None # Set to an actual client for testing Qdrant part standalone
    # if test_qdrant_client is None:
    #     logger.warning("Standalone test: Qdrant client not initialized, Qdrant updates will be skipped.")

    # Create a dummy args object for the standalone test
    class DummyArgs:
        def __init__(self):
            self.project = "runs/track/linker_test_exp" # Corresponds to project_dir
            self.name = "" # No sub-name for this test, so dataset goes into project_dir/dataset
    
    dummy_args = DummyArgs()

    output_path, metrics = link_multimodal_data(
        video_dataset_path=dummy_video_dataset_path, 
        audio_segments_path=dummy_audio_segments_path, 
        qdrant_client=test_qdrant_client,
        args=dummy_args
    )

    if output_path:
        logger.info(f"Standalone test successful. Enriched data at: {output_path}")
        logger.info(f"Standalone test metrics: {metrics}")
    else:
        logger.error(f"Standalone test failed. Metrics: {metrics}") 