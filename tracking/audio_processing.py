"""
Handles the audio processing pipeline: extraction, diarization, and transcription.
"""

import argparse
from pathlib import Path
import logging
import os
import uuid # For generating unique IDs for audio segments
import ffmpeg
import torch # Added for pyannote
from pyannote.audio import Pipeline # Added for pyannote
import soundfile as sf # Added for loading audio for ASR
import nemo.collections.asr as nemo_asr # Added for Parakeet ASR
import numpy as np # Added for audio slicing

from dotenv import load_dotenv
load_dotenv()

# Import a a f
from tracking.vectorize import embed_text, add_audio_transcript_embedding

# Ensure TOKENIZERS_PARALLELISM is set if using Hugging Face transformers for transcription/diarization
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

# Global variables for pipelines to avoid reloading
DIARIZATION_PIPELINE = None
ASR_PIPELINE = None
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2" # Using this model as decided

def get_diarization_pipeline(hf_token: str = None):
    global DIARIZATION_PIPELINE
    if DIARIZATION_PIPELINE is None:
        logger.info(f"Loading speaker diarization pipeline pyannote/speaker-diarization-3.1...")
        effective_token = hf_token or os.getenv("HF_TOKEN")
        if not effective_token:
            logger.warning("Hugging Face token not provided. Diarization pipeline will not be loaded.")
        else:
            try:
                DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=effective_token
                )
                if torch.cuda.is_available():
                    logger.info("Moving diarization pipeline to GPU.")
                    DIARIZATION_PIPELINE.to(torch.device("cuda"))
                elif torch.backends.mps.is_available():
                    logger.info("Moving diarization pipeline to MPS (Apple Silicon GPU).")
                    DIARIZATION_PIPELINE.to(torch.device("mps"))
                else:
                    logger.info("Diarization pipeline will run on CPU.")
                logger.info("Speaker diarization pipeline loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading pyannote.audio diarization pipeline: {e}", exc_info=True)
                DIARIZATION_PIPELINE = None
    return DIARIZATION_PIPELINE

def get_asr_pipeline():
    global ASR_PIPELINE
    if ASR_PIPELINE is None:
        logger.info(f"Loading ASR pipeline: {ASR_MODEL_NAME}...")
        try:
            ASR_PIPELINE = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
            if torch.cuda.is_available():
                logger.info("Moving ASR pipeline to GPU.")
                ASR_PIPELINE.to(torch.device("cuda"))
            elif torch.backends.mps.is_available():
                logger.info("Moving ASR pipeline to MPS (Apple Silicon GPU).")
                ASR_PIPELINE.to(torch.device("mps")) 
            else:
                logger.info("ASR pipeline will run on CPU.")
            logger.info(f"ASR pipeline {ASR_MODEL_NAME} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading ASR pipeline {ASR_MODEL_NAME}: {e}", exc_info=True)
            ASR_PIPELINE = None
    return ASR_PIPELINE

def extract_audio(video_source_path: str, output_audio_path: str) -> bool:
    logger.info(f"Attempting to extract audio from '{video_source_path}' to '{output_audio_path}'")
    try:
        (
            ffmpeg
            .input(video_source_path)
            .output(output_audio_path, acodec='pcm_s16le', ar='16000', ac=1)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        logger.info(f"Successfully extracted audio to '{output_audio_path}'")
        return True
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
        logger.error(f"ffmpeg error during audio extraction for '{video_source_path}':\n{error_message}", exc_info=False)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio extraction for '{video_source_path}': {e}", exc_info=True)
        return False

def diarize_audio(audio_path: str, hf_token: str = None) -> list:
    logger.info(f"Performing speaker diarization on '{audio_path}'")
    pipeline = get_diarization_pipeline(hf_token=hf_token)
    if pipeline is None:
        logger.error("Diarization pipeline is not available. Cannot proceed.")
        return []
    try:
        diarization_result = pipeline(audio_path)
        diarized_segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segment = {
                'speaker_label': speaker,
                'start_time': round(turn.start, 3),
                'end_time': round(turn.end, 3)
            }
            diarized_segments.append(segment)
            logger.debug(f"Diarized segment: Speaker {speaker} from {segment['start_time']:.2f}s to {segment['end_time']:.2f}s")
        logger.info(f"Diarization complete. Found {len(diarized_segments)} segments.")
        return diarized_segments
    except Exception as e:
        logger.error(f"Error during diarization for '{audio_path}': {e}", exc_info=True)
        return []

def transcribe_audio_segments(audio_path: str, segments: list, audio_output_dir: Path) -> list:
    logger.info(f"Attempting to transcribe {len(segments)} audio segments from '{audio_path}' using {ASR_MODEL_NAME}.")
    asr_model = get_asr_pipeline()
    if asr_model is None:
        logger.error(f"ASR pipeline {ASR_MODEL_NAME} is not available. Skipping transcription.")
        for i, segment in enumerate(segments):
            segment['transcript_text'] = "ASR model unavailable - placeholder transcript."
            segment['audio_segment_unique_id'] = str(uuid.uuid4())
        return segments
    try:
        full_audio, sample_rate = sf.read(audio_path, dtype='float32')
        if sample_rate != 16000:
            logger.warning(f"Audio sample rate is {sample_rate}Hz, but ASR model expects 16000Hz.")
        temp_segment_files = []
        for i, segment in enumerate(segments):
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            speaker_label = segment.get('speaker_label', 'UNKNOWN_SPEAKER')
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            if start_sample >= end_sample or start_sample >= len(full_audio) or end_sample <= 0:
                logger.warning(f"Segment {i+1} for speaker {speaker_label} ({start_time:.2f}s - {end_time:.2f}s) has invalid time range. Skipping.")
                segment['transcript_text'] = ""
                segment['audio_segment_unique_id'] = str(uuid.uuid4())
                continue
            start_sample = max(0, start_sample)
            end_sample = min(len(full_audio), end_sample)
            audio_slice = full_audio[start_sample:end_sample]
            if len(audio_slice) == 0:
                logger.warning(f"Segment {i+1} for speaker {speaker_label} ({start_time:.2f}s - {end_time:.2f}s) empty slice. Skipping.")
                segment['transcript_text'] = ""
                segment['audio_segment_unique_id'] = str(uuid.uuid4())
                continue
            segment_unique_id = segment.get('audio_segment_unique_id') # Use existing if populated earlier
            if not segment_unique_id: # Generate if transcribe_audio_segments is the first place it's made
                 segment_unique_id = str(uuid.uuid4())
                 segment['audio_segment_unique_id'] = segment_unique_id
            
            temp_audio_path = audio_output_dir / f"temp_segment_{segment_unique_id}.wav"
            try:
                sf.write(str(temp_audio_path), audio_slice, sample_rate)
                temp_segment_files.append(temp_audio_path)
                transcription_result = asr_model.transcribe([str(temp_audio_path)])
                hyp_text = ""
                if transcription_result and isinstance(transcription_result, list) and len(transcription_result) > 0:
                    if hasattr(transcription_result[0], 'text'):
                        hyp_text = transcription_result[0].text
                    elif isinstance(transcription_result[0], str):
                        hyp_text = transcription_result[0]
                segment['transcript_text'] = hyp_text.strip()
                logger.debug(f"  Segment {i+1} ({segment_unique_id}): Transcribed: '{hyp_text[:50]}...'")
            except Exception as e_slice:
                logger.error(f"Error transcribing segment {i+1} ({start_time:.2f}s - {end_time:.2f}s): {e_slice}", exc_info=True)
                segment['transcript_text'] = "Transcription error for this segment."
    except Exception as e_load_full:
        logger.error(f"Error loading/processing full audio '{audio_path}': {e_load_full}", exc_info=True)
        for segment in segments:
            segment['transcript_text'] = "Error loading base audio for transcription."
            if 'audio_segment_unique_id' not in segment:
                 segment['audio_segment_unique_id'] = str(uuid.uuid4())
    finally:
        for temp_file in temp_segment_files:
            try:
                os.remove(temp_file)
                logger.debug(f"Deleted temp: {temp_file}")
            except OSError as e_del:
                logger.warning(f"Could not delete temp file {temp_file}: {e_del}")
    logger.info(f"Transcription of {len(segments)} segments complete.")
    return segments

def process_audio(args):
    logger.info(f"Starting audio processing pipeline for source: {args.source}")
    audio_processing_metrics = {}
    audio_output_paths = {}
    hf_auth_token = getattr(args, 'hf_token', os.getenv('HF_TOKEN'))

    audio_output_dir = Path(args.project) / args.name / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(args.source).stem
    extracted_audio_wav_path = audio_output_dir / f"{base_name}_extracted.wav"
    audio_output_paths['extracted_wav'] = str(extracted_audio_wav_path)

    logger.info("--- Audio Extraction Stage ---")
    if not extract_audio(args.source, str(extracted_audio_wav_path)):
        logger.error("Audio extraction failed. Cannot proceed.")
        return [], audio_processing_metrics, audio_output_paths

    logger.info("--- Speaker Diarization Stage ---")
    diarized_segments = diarize_audio(str(extracted_audio_wav_path), hf_token=hf_auth_token)
    if not diarized_segments:
        logger.warning("No diarized segments found.")
        # Ensure unique IDs are added even if diarization is empty, before transcription attempts to use them
        for seg in diarized_segments: # This loop won't run if diarized_segments is empty
            if 'audio_segment_unique_id' not in seg:
                seg['audio_segment_unique_id'] = str(uuid.uuid4())
    else: # If diarization produced segments, ensure they all have IDs before transcription
        for seg in diarized_segments:
            if 'audio_segment_unique_id' not in seg:
                seg['audio_segment_unique_id'] = str(uuid.uuid4())

    logger.info("--- Transcription Stage ---")
    processed_audio_segments = transcribe_audio_segments(str(extracted_audio_wav_path), diarized_segments, audio_output_dir)

    logger.info("--- Audio Transcript Embedding Stage ---")
    segments_embedded_count = 0
    for segment in processed_audio_segments:
        transcript = segment.get('transcript_text', '').strip()
        segment_id = segment.get('audio_segment_unique_id') # Should exist from transcription stage

        if not transcript:
            logger.debug(f"Segment {segment_id} has empty transcript. Skipping embedding.")
            segment['transcript_vector_id'] = None # Explicitly mark as not embedded
            continue
        if not segment_id:
            logger.warning(f"Segment missing 'audio_segment_unique_id'. Transcript: '{transcript[:30]}...'. Skipping embedding.")
            segment['transcript_vector_id'] = None
            continue
        
        try:
            text_embedding = embed_text(transcript) # From tracking.vectorize
            if text_embedding is not None:
                qdrant_point_id = add_audio_transcript_embedding(
                    embedding=text_embedding,
                    audio_segment_unique_id=segment_id,
                    speaker_label=segment.get('speaker_label', 'UNKNOWN'),
                    start_time_seconds=segment.get('start_time', 0.0),
                    end_time_seconds=segment.get('end_time', 0.0),
                    transcript_text=transcript
                )
                if qdrant_point_id:
                    segment['transcript_vector_id'] = qdrant_point_id
                    segments_embedded_count += 1
                    logger.debug(f"Segment {segment_id} transcript embedded and stored with Qdrant ID: {qdrant_point_id}")
                else:
                    logger.error(f"Failed to store transcript embedding for segment {segment_id} in Qdrant.")
                    segment['transcript_vector_id'] = None
            else:
                logger.error(f"Failed to generate text embedding for segment {segment_id}. Transcript: '{transcript[:30]}...'")
                segment['transcript_vector_id'] = None
        except Exception as e_embed:
            logger.error(f"Error during embedding or Qdrant storage for segment {segment_id}: {e_embed}", exc_info=True)
            segment['transcript_vector_id'] = None

    logger.info(f"Audio transcript embedding stage completed. {segments_embedded_count}/{len(processed_audio_segments)} segments embedded.")
    audio_processing_metrics['segments_transcripts_embedded'] = segments_embedded_count

    audio_segments_json_path = audio_output_dir / f"{base_name}_audio_segments.json"
    try:
        with open(audio_segments_json_path, 'w') as f:
            import json
            json.dump(processed_audio_segments, f, indent=2)
        logger.info(f"Detailed audio segments data saved to: {audio_segments_json_path}")
        audio_output_paths['audio_segments_json'] = str(audio_segments_json_path)
    except Exception as e:
        logger.error(f"Error saving audio segments JSON: {e}", exc_info=True)

    audio_processing_metrics['total_segments_diarized'] = len(diarized_segments)
    audio_processing_metrics['total_segments_transcribed'] = len(processed_audio_segments) # This count is before embedding attempted

    logger.info(f"Audio processing pipeline completed. Produced {len(processed_audio_segments)} segments.")
    return processed_audio_segments, audio_processing_metrics, audio_output_paths


def parse_audio_proc_args():
    parser = argparse.ArgumentParser(description="Audio Processing Module")
    parser.add_argument('--source', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--project', type=Path, required=True, help='Base directory for saving results')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN'), help='Hugging Face authentication token. Can also be set via HF_TOKEN env var.')
    return parser.parse_args()

if __name__ == '__main__':
    print("Running audio_processing.py as a standalone script.")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("nemo_logging").setLevel(logging.WARNING) 
    
    args = parse_audio_proc_args()
    if args.hf_token is None:
        logger.warning("Hugging Face token not provided for standalone run. Diarization might fail.")

    if not Path(args.source).exists():
         logger.error(f"Source video file not found: {args.source}.")
    else:
        logger.info(f"Starting standalone audio processing test for source: {args.source}")
        # Ensure vectorize.py has initialized its model if not already done via imports elsewhere
        # For standalone, it's good practice if vectorize.py initializes its model on import or first use.
        segments, metrics, paths = process_audio(args)
        logger.info("Standalone audio processing completed.")
        logger.info(f"Processed segments: {len(segments)}")
        if segments:
            logger.info(f"First segment example: {segments[0]}")
        logger.info(f"Metrics: {json.dumps(metrics, indent=2) if metrics else 'No metrics'}")
        logger.info(f"Output paths: {paths}") 