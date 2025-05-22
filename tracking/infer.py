import os
import argparse
import logging
import base64
import io
from typing import List, Dict, Optional, Any
import datetime
import json

import openai
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase, basic_auth
from minio import Minio
from minio.error import S3Error

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration for Image Storage vs Captioning ---
USE_CAPTIONING = os.getenv("USE_CAPTIONING", "False").lower() == "true"
print(f"[infer.py] USE_CAPTIONING mode: {USE_CAPTIONING}")

# --- Constants & Model Config --- (Shared with vectorize.py logic)
ENTITY_COLLECTION_NAME = "entities"
FRAME_COLLECTION_NAME = "frames"
AUDIO_COLLECTION_NAME = "audio_transcripts"
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL")

SUPPORTED_EMBEDDING_MODELS = {
    "clip-ViT-B-32": {"dimension": 512, "type": "sentence_transformer"},
    "clip-ViT-L-14": {"dimension": 768, "type": "sentence_transformer"}
    # "google/siglip2-so400m-patch16-512": {"dimension": 1152, "type": "siglip"}
}
DEFAULT_MODEL_NAME = "clip-ViT-B-32"
MODEL_NAME_FROM_ENV = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)

if MODEL_NAME_FROM_ENV not in SUPPORTED_EMBEDDING_MODELS:
    print(f"Warning: EMBEDDING_MODEL_NAME '{MODEL_NAME_FROM_ENV}' is not in SUPPORTED_EMBEDDING_MODELS. Falling back to default: '{DEFAULT_MODEL_NAME}'.")
    MODEL_NAME = DEFAULT_MODEL_NAME
else:
    MODEL_NAME = MODEL_NAME_FROM_ENV

MODEL_CONFIG = SUPPORTED_EMBEDDING_MODELS[MODEL_NAME]
EMBED_DIM = MODEL_CONFIG["dimension"] # Not strictly needed in infer.py unless creating collections, but good for consistency
MODEL_TYPE = MODEL_CONFIG["type"]

# --- Constants for Image Retrieval Strategy ---
TARGET_ENTITY_TOP_K = 3
TARGET_FRAME_TOP_K = 2
RETRIEVAL_MULTIPLIER = 2 # Fetch this many times the target for initial selection
MIN_SIMILARITY_THRESHOLD = 0.20 # Minimum score to consider an image relevant

TARGET_AUDIO_TOP_K = 3 # How many top audio segments to retrieve
AUDIO_RETRIEVAL_MULTIPLIER = 2 # Multiplier for audio retrieval
MIN_AUDIO_SIMILARITY_THRESHOLD = 0.25 # Minimum score for audio segment relevance

print(f"[infer.py] Using embedding model: {MODEL_NAME} (Type: {MODEL_TYPE})")

# --- Logging Configuration ---
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
logger = logging.getLogger(__name__)

# --- MinIO Configuration (mirroring vectorize.py for consistency) ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "boxmot-images")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "False").lower() == "true"

# --- Global Clients and Models (to be initialized) ---
embedding_model: Optional[SentenceTransformer] = None
qdrant_client: Optional[QdrantClient] = None
neo4j_driver: Optional[GraphDatabase.driver] = None  # type: ignore
openai_client: Optional[openai.OpenAI] = None
minio_client_infer: Optional[Minio] = None # Separate MinIO client instance for infer.py


# --- Core Components Initialization ---
def initialize_embedding_model():
    """Loads the configured SentenceTransformer embedding model."""
    global embedding_model
    if embedding_model:
        logger.info(f"SentenceTransformer model ({MODEL_NAME}) already initialized.")
        return
    
    if MODEL_TYPE != "sentence_transformer":
        logger.error(f"[infer.py] Unsupported MODEL_TYPE '{MODEL_TYPE}' for {MODEL_NAME} in infer.py. Only 'sentence_transformer' is currently set up here.")
        raise ValueError(f"Unsupported MODEL_TYPE for infer.py: {MODEL_TYPE}")
    try:
        logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
        embedding_model = SentenceTransformer(MODEL_NAME, device='cpu') # Adjust device as needed
        embedding_model.eval()
        logger.info(f"SentenceTransformer model ({MODEL_NAME}) loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model ({MODEL_NAME}): {e}", exc_info=True)
        raise


def initialize_qdrant_client(host="localhost", port=6333):
    """Initializes the Qdrant client."""
    global qdrant_client
    if qdrant_client:
        logger.info("Qdrant client already initialized.")
        return
    try:
        logger.info(f"Connecting to Qdrant at {host}:{port}")
        qdrant_client = QdrantClient(host=host, port=port)
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if ENTITY_COLLECTION_NAME not in collection_names:
            logger.warning(
                f"Qdrant collection '{ENTITY_COLLECTION_NAME}' not found. Searches may fail or collection might be created with default dimension if not already handled by vectorize.py."
            )
        if FRAME_COLLECTION_NAME not in collection_names:
            logger.warning(
                f"Qdrant collection '{FRAME_COLLECTION_NAME}' not found. Searches may fail or collection might be created with default dimension if not already handled by vectorize.py."
            )
        logger.info("Qdrant client initialized and collections checked.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise


def initialize_neo4j_driver(uri, user, password):
    """Initializes the Neo4j driver."""
    global neo4j_driver
    if neo4j_driver:
        logger.info("Neo4j driver already initialized.")
        return
    try:
        logger.info(f"Connecting to Neo4j at {uri}")
        neo4j_driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        neo4j_driver.verify_connectivity()
        logger.info("Neo4j driver initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j driver: {e}")
        raise


def initialize_openai_client():
    """Initializes the OpenAI client."""
    global openai_client
    if openai_client:
        logger.info("OpenAI client already initialized.")
        return
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY is required for OpenAI client.")
    try:
        openai_client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise


def initialize_minio_client_infer():
    """Initializes the MinIO client for inference tasks."""
    global minio_client_infer
    if minio_client_infer:
        logger.info("MinIO client (infer) already initialized.")
        return
    try:
        logger.info(f"Initializing MinIO client (infer) for endpoint: {MINIO_ENDPOINT}, bucket: {MINIO_BUCKET_NAME}")
        minio_client_infer = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_USE_SSL
        )
        # Verify bucket exists (optional, as vectorize.py should create it)
        if not minio_client_infer.bucket_exists(MINIO_BUCKET_NAME):
            logger.warning(f"MinIO bucket '{MINIO_BUCKET_NAME}' does not exist. Image retrieval will fail. Ensure vectorize.py has run and created the bucket.")
        else:
            logger.info(f"MinIO client (infer) initialized and bucket '{MINIO_BUCKET_NAME}' confirmed.")
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client (infer): {e}", exc_info=True)
        minio_client_infer = None # Ensure client is None if initialization fails
        raise # Re-raise to signal a critical failure if MinIO is essential


def download_image_from_minio(object_name: str, logger_instance: logging.Logger) -> Optional[bytes]:
    """Downloads an image from MinIO and returns its bytes."""
    if not minio_client_infer:
        logger_instance.error("[download_image_from_minio] MinIO client (infer) not initialized. Cannot download image.")
        return None
    try:
        response = minio_client_infer.get_object(MINIO_BUCKET_NAME, object_name)
        image_bytes = response.read()
        logger_instance.info(f"[download_image_from_minio] Successfully downloaded {object_name} from MinIO bucket {MINIO_BUCKET_NAME}.")
        return image_bytes
    except S3Error as exc:
        logger_instance.error(f"[download_image_from_minio] Error downloading {object_name} from MinIO: {exc}", exc_info=True)
        return None
    finally:
        if 'response' in locals() and response: # type: ignore
            response.close()
            response.release_conn()


# --- Embedding and Similarity Functions ---
def embed_text(text: str) -> Optional[np.ndarray]:
    """Embeds text using the initialized SentenceTransformer model."""
    if not embedding_model:
        logger.error("SentenceTransformer model not initialized. Cannot embed text.")
        return None
    if MODEL_TYPE != "sentence_transformer":
        raise NotImplementedError(f"embed_text not implemented for MODEL_TYPE {MODEL_TYPE}")
    try:
        # SentenceTransformer.encode() expects a list of sentences
        text_embedding = embedding_model.encode([text])
        embedding_to_log = text_embedding[0].astype(np.float32).flatten()
        logger.info(f"[embed_text] Generated embedding for '{text[:30]}...' - Shape: {embedding_to_log.shape}, Mean: {embedding_to_log.mean():.4f}, Std: {embedding_to_log.std():.4f}, Min: {embedding_to_log.min():.4f}, Max: {embedding_to_log.max():.4f}")
        if np.all(embedding_to_log == 0):
            logger.warning("[embed_text] Warning: Generated embedding is all zeros.")
        return embedding_to_log
    except Exception as e:
        logger.error(f"Error embedding text '{text[:50]}...' with {MODEL_NAME}: {e}", exc_info=True)
        return None


# --- Helper Functions ---
def pil_to_base64(image: Image.Image, format="PNG") -> str:
    """Converts a PIL Image to a base64 encoded string with data URI."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def retrieve_similar_content(
    query_text: str, # Keep query_text for potential future use or logging, though embedding is primary
    query_embedding: np.ndarray,
    qdrant_client_instance: QdrantClient,
    logger_instance: logging.Logger
) -> List[Dict[str, Any]]:
    content_type = "captions" if USE_CAPTIONING else "images"
    logger_instance.info(f"[retrieve_similar_content] Entered with query: '{query_text[:50]}...'. Mode: {content_type}. Strategy: Entities (target {TARGET_ENTITY_TOP_K}), Frames (target {TARGET_FRAME_TOP_K}), Threshold ({MIN_SIMILARITY_THRESHOLD})")
    if not qdrant_client_instance:
        logger_instance.error(f"[retrieve_similar_content] Qdrant client not initialized. Cannot retrieve {content_type}.")
        return []

    if query_embedding is None:
        logger_instance.warning(f"[retrieve_similar_content] Invalid query_embedding (None) for query: {query_text}")
        return []
    logger_instance.info(f"[retrieve_similar_content] Using pre-computed query embedding (shape: {query_embedding.shape}) from {MODEL_NAME}.")

    selected_entity_results: List[Dict[str, Any]] = []
    selected_frame_results: List[Dict[str, Any]] = []

    # --- Retrieve and select ENTITY content ---
    initial_entity_retrieve_limit = TARGET_ENTITY_TOP_K * RETRIEVAL_MULTIPLIER
    logger_instance.info(f"[retrieve_similar_content] Searching '{ENTITY_COLLECTION_NAME}' collection with initial limit={initial_entity_retrieve_limit}.")
    try:
        entity_hits_result = qdrant_client_instance.query_points(
            collection_name=ENTITY_COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=initial_entity_retrieve_limit,
            score_threshold=MIN_SIMILARITY_THRESHOLD, # Qdrant can filter by threshold
            with_payload=True,
        )
        logger_instance.info(f"[retrieve_similar_content] Raw Qdrant entity_hits_result for query '{query_text[:30]}...': Count: {len(entity_hits_result.points)}")
        if len(entity_hits_result.points) > 0:
            for i, hit in enumerate(entity_hits_result.points[:3]): # Log first 3 raw hits
                 logger_instance.info(f"[retrieve_similar_content] Raw Entity Hit {i+1}: ID={hit.id}, Score={hit.score:.4f}, Payload Keys={list(hit.payload.keys()) if hit.payload else 'No Payload'}")

        entity_hits = entity_hits_result.points
        logger_instance.info(f"[retrieve_similar_content] Found {len(entity_hits)} hits in '{ENTITY_COLLECTION_NAME}' (after Qdrant threshold).")
        
        processed_entity_hits = []
        for hit in entity_hits: # Already sorted by score by Qdrant
            if hit.score >= MIN_SIMILARITY_THRESHOLD: # Redundant if score_threshold in query_points works as expected, but good as a safeguard
                payload = hit.payload if hit.payload else {}
                content_data = None
                content_text = None
                
                if USE_CAPTIONING:
                    # Get caption text instead of image
                    caption_text = payload.get("image_caption")
                    if caption_text:
                        content_text = caption_text
                        logger_instance.info(f"[retrieve_similar_content] Entity hit {hit.id} (score: {hit.score:.4f}) has caption: '{caption_text[:50]}{'...' if len(caption_text) > 50 else ''}'")
                    else:
                        logger_instance.info(f"[retrieve_similar_content] Entity hit {hit.id} (score: {hit.score:.4f}) had no 'image_caption' in payload.")
                else:
                    # Get image from MinIO (original behavior)
                    minio_object_id = payload.get("image_minio_id")
                    if minio_object_id:
                        image_bytes = download_image_from_minio(minio_object_id, logger_instance)
                        if image_bytes:
                            try:
                                content_data = Image.open(io.BytesIO(image_bytes))
                                logger_instance.info(f"[retrieve_similar_content] Entity hit {hit.id} (score: {hit.score:.4f}) loaded image from MinIO: {minio_object_id}")
                            except Exception as e_pil:
                                logger_instance.error(f"[retrieve_similar_content] Entity hit {hit.id} (MinIO ID: {minio_object_id}) - Failed to create PIL Image from downloaded bytes: {e_pil}")
                    else:
                        logger_instance.info(f"[retrieve_similar_content] Entity hit {hit.id} (score: {hit.score:.4f}) had no 'image_minio_id' in payload.")
                
                if content_data or content_text:
                    processed_entity_hits.append({
                        "vector_id": hit.id,
                        "type": "entity",
                        "entity_id": payload.get("entity_id", payload.get("id")),
                        "class_name": payload.get("class_name", "N/A"),
                        "confidence": payload.get("confidence"),
                        "image_data": content_data,  # PIL Image or None
                        "caption_text": content_text,  # Caption text or None
                        "score": hit.score,
                    })
                else:
                    logger_instance.info(f"[retrieve_similar_content] Entity hit {hit.id} (score: {hit.score:.4f}) had no content data (image or caption).")
            else:
                 logger_instance.info(f"[retrieve_similar_content] Entity hit {hit.id} (score: {hit.score:.4f}) below explicit threshold {MIN_SIMILARITY_THRESHOLD}, discarding.")


        selected_entity_results = processed_entity_hits[:TARGET_ENTITY_TOP_K]
        logger_instance.info(f"[retrieve_similar_content] Selected {len(selected_entity_results)} entity {content_type} after filtering and capping at {TARGET_ENTITY_TOP_K}.")

    except Exception as e:
        logger_instance.error(f"[retrieve_similar_content] Error searching Qdrant collection '{ENTITY_COLLECTION_NAME}': {e}", exc_info=True)

    # --- Retrieve and select FRAME content ---
    initial_frame_retrieve_limit = TARGET_FRAME_TOP_K * RETRIEVAL_MULTIPLIER
    logger_instance.info(f"[retrieve_similar_content] Searching '{FRAME_COLLECTION_NAME}' collection with initial limit={initial_frame_retrieve_limit}.")
    try:
        frame_hits_result = qdrant_client_instance.query_points(
            collection_name=FRAME_COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=initial_frame_retrieve_limit,
            score_threshold=MIN_SIMILARITY_THRESHOLD, # Qdrant can filter by threshold
            with_payload=True,
        )
        logger_instance.info(f"[retrieve_similar_content] Raw Qdrant frame_hits_result for query '{query_text[:30]}...': Count: {len(frame_hits_result.points)}")
        if len(frame_hits_result.points) > 0:
            for i, hit in enumerate(frame_hits_result.points[:3]): # Log first 3 raw hits
                 logger_instance.info(f"[retrieve_similar_content] Raw Frame Hit {i+1}: ID={hit.id}, Score={hit.score:.4f}, Payload Keys={list(hit.payload.keys()) if hit.payload else 'No Payload'}")

        frame_hits = frame_hits_result.points
        logger_instance.info(f"[retrieve_similar_content] Found {len(frame_hits)} hits in '{FRAME_COLLECTION_NAME}' (after Qdrant threshold).")

        processed_frame_hits = []
        for hit in frame_hits: # Already sorted by score
            if hit.score >= MIN_SIMILARITY_THRESHOLD: # Safeguard
                payload = hit.payload if hit.payload else {}
                content_data = None
                content_text = None
                
                if USE_CAPTIONING:
                    # Get caption text instead of image
                    caption_text = payload.get("image_caption")
                    if caption_text:
                        content_text = caption_text
                        logger_instance.info(f"[retrieve_similar_content] Frame hit {hit.id} (score: {hit.score:.4f}) has caption: '{caption_text[:50]}{'...' if len(caption_text) > 50 else ''}'")
                    else:
                        logger_instance.info(f"[retrieve_similar_content] Frame hit {hit.id} (score: {hit.score:.4f}) had no 'image_caption' in payload.")
                else:
                    # Get image from MinIO (original behavior)
                    minio_object_id = payload.get("minio_image_path")
                    if minio_object_id:
                        image_bytes = download_image_from_minio(minio_object_id, logger_instance)
                        if image_bytes:
                            try:
                                content_data = Image.open(io.BytesIO(image_bytes))
                                logger_instance.info(f"[retrieve_similar_content] Frame hit {hit.id} (score: {hit.score:.4f}) loaded image from MinIO: {minio_object_id}")
                            except Exception as e_pil:
                                logger_instance.error(f"[retrieve_similar_content] Frame hit {hit.id} (MinIO ID: {minio_object_id}) - Failed to create PIL Image from downloaded bytes: {e_pil}")
                    else:
                        logger_instance.info(f"[retrieve_similar_content] Frame hit {hit.id} (score: {hit.score:.4f}) had no 'minio_image_path' in payload.")

                if content_data or content_text:
                    processed_frame_hits.append({
                        "vector_id": hit.id,
                        "type": "frame",
                        "entity_id": f"frame_{payload.get('frame_idx', 'unknown')}", # Consistent IDing
                        "class_name": "Frame",
                        "confidence": None, # Frames don't have detection confidence
                        "image_data": content_data,  # PIL Image or None
                        "caption_text": content_text,  # Caption text or None
                        "score": hit.score,
                    })
                else:
                    logger_instance.info(f"[retrieve_similar_content] Frame hit {hit.id} (score: {hit.score:.4f}) had no content data (image or caption).")
            else:
                logger_instance.info(f"[retrieve_similar_content] Frame hit {hit.id} (score: {hit.score:.4f}) below explicit threshold {MIN_SIMILARITY_THRESHOLD}, discarding.")
        
        selected_frame_results = processed_frame_hits[:TARGET_FRAME_TOP_K]
        logger_instance.info(f"[retrieve_similar_content] Selected {len(selected_frame_results)} frame {content_type} after filtering and capping at {TARGET_FRAME_TOP_K}.")

    except Exception as e:
        logger_instance.error(f"[retrieve_similar_content] Error searching Qdrant collection '{FRAME_COLLECTION_NAME}': {e}", exc_info=True)
    
    # Combine results and sort by score
    final_results = selected_entity_results + selected_frame_results
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    logger_instance.info(f"[retrieve_similar_content] Returning {len(final_results)} total {content_type} after independent selection and final sort.")
    return final_results


def retrieve_relevant_audio_transcripts(
    query_embedding: np.ndarray,
    qdrant_client_instance: QdrantClient,
    logger_instance: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant audio transcript segments from Qdrant based on the query embedding.
    """
    logger_instance.info(f"[retrieve_relevant_audio_transcripts] Entered. Strategy: Top K={TARGET_AUDIO_TOP_K}, Multiplier={AUDIO_RETRIEVAL_MULTIPLIER}, Threshold={MIN_AUDIO_SIMILARITY_THRESHOLD}")
    if not qdrant_client_instance:
        logger_instance.error("[retrieve_relevant_audio_transcripts] Qdrant client not initialized. Cannot retrieve audio transcripts.")
        return []

    if query_embedding is None:
        logger_instance.warning("[retrieve_relevant_audio_transcripts] Invalid query_embedding (None).")
        return []

    initial_audio_retrieve_limit = TARGET_AUDIO_TOP_K * AUDIO_RETRIEVAL_MULTIPLIER
    logger_instance.info(f"[retrieve_relevant_audio_transcripts] Searching '{AUDIO_COLLECTION_NAME}' collection with initial limit={initial_audio_retrieve_limit}.")
    
    retrieved_audio_segments: List[Dict[str, Any]] = []

    try:
        audio_hits_result = qdrant_client_instance.query_points(
            collection_name=AUDIO_COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=initial_audio_retrieve_limit,
            score_threshold=MIN_AUDIO_SIMILARITY_THRESHOLD,
            with_payload=True,
        )
        logger_instance.info(f"[retrieve_relevant_audio_transcripts] Raw Qdrant audio_hits_result: Count: {len(audio_hits_result.points)}")

        audio_hits = audio_hits_result.points
        logger_instance.info(f"[retrieve_relevant_audio_transcripts] Found {len(audio_hits)} hits in '{AUDIO_COLLECTION_NAME}' (after Qdrant threshold).")

        for hit in audio_hits: # Already sorted by score
            if hit.score >= MIN_AUDIO_SIMILARITY_THRESHOLD: # Safeguard
                payload = hit.payload if hit.payload else {}
                transcript_text = payload.get("transcript_text")
                
                if transcript_text:
                    retrieved_audio_segments.append({
                        "audio_segment_unique_id": payload.get("audio_segment_unique_id", hit.id),
                        "transcript_text": transcript_text,
                        "speaker_label": payload.get("speaker_label", "N/A"),
                        "start_time_seconds": payload.get("start_time_seconds"),
                        "end_time_seconds": payload.get("end_time_seconds"),
                        "score": hit.score,
                    })
                else:
                    logger_instance.info(f"[retrieve_relevant_audio_transcripts] Audio hit {hit.id} (score: {hit.score:.4f}) had no transcript_text.")
            else:
                logger_instance.info(f"[retrieve_relevant_audio_transcripts] Audio hit {hit.id} (score: {hit.score:.4f}) below explicit threshold {MIN_AUDIO_SIMILARITY_THRESHOLD}, discarding.")
        
        # Sort by score (Qdrant already does this, but an explicit sort after custom processing is good practice)
        retrieved_audio_segments.sort(key=lambda x: x["score"], reverse=True)
        # Select top K
        selected_audio_segments = retrieved_audio_segments[:TARGET_AUDIO_TOP_K]
        logger_instance.info(f"[retrieve_relevant_audio_transcripts] Selected {len(selected_audio_segments)} audio segments after filtering and capping at {TARGET_AUDIO_TOP_K}.")

    except Exception as e:
        logger_instance.error(f"[retrieve_relevant_audio_transcripts] Error searching Qdrant collection '{AUDIO_COLLECTION_NAME}': {e}", exc_info=True)
        return [] # Return empty list on error

    return selected_audio_segments


def retrieve_graph_context(entity_ids: List[str]) -> str:
    if not neo4j_driver:
        logger.error("Neo4j driver not initialized. Cannot retrieve graph context.")
        return "Graph database connection is not available."
    if not entity_ids:
        return "No entity IDs provided to retrieve graph context."

    all_summaries: List[str] = []
    processed_entity_ids = set() # To avoid processing the same entity multiple times if duplicated in input

    with neo4j_driver.session() as session:
        # 1. Individual Entity Summaries
        for entity_id_str_original in entity_ids:
            entity_id_str = str(entity_id_str_original) # Ensure string
            if entity_id_str in processed_entity_ids:
                continue
            processed_entity_ids.add(entity_id_str)

            try:
                query_individual = """
                MATCH (e:Entity {entity_id: $entity_id})
                OPTIONAL MATCH (e)-[:DETECTED_IN]->(f:Frame)
                WITH e, collect(DISTINCT f.frame_idx) AS frame_indices_prop
                OPTIONAL MATCH (e)-[r]-(other_e:Entity)
                WHERE NOT type(r) IN ['DETECTED_IN', 'NEXT', 'PREV'] AND e.entity_id <> other_e.entity_id
                WITH e, frame_indices_prop,
                     collect(DISTINCT {type: type(r), target_id: other_e.entity_id, target_name: COALESCE(other_e.name, 'Entity ' + other_e.entity_id)}) AS relations_prop
                RETURN e.name AS entity_name_prop,
                       e.entity_id AS id_prop,
                       frame_indices_prop,
                       relations_prop
                """
                result = session.run(query_individual, entity_id=entity_id_str)
                record = result.single()

                if record:
                    entity_name = record.get("entity_name_prop", f"Entity {record['id_prop']}")
                    frames = sorted([idx for idx in record.get("frame_indices_prop", []) if idx is not None])
                    summary = f"{entity_name} (ID: {record['id_prop']})"
                    if frames:
                        if len(frames) > 5:
                            summary += f" appears in frames {frames[0]}...{frames[-1]} (total {len(frames)} frames)."
                        else:
                            summary += f" appears in frames: {', '.join(map(str, frames))}."
                    else:
                        summary += " has no recorded frame appearances."
                    
                    relations = [rel for rel in record.get("relations_prop", []) if rel and rel.get("type") and rel.get("target_id")]
                    # Filter out relations to entities NOT in the initially provided entity_ids list for this individual summary part
                    # to avoid redundancy if inter-entity relationships are handled separately and more explicitly.
                    # However, for a general description of an entity, showing all its direct non-frame, non-sequence relations is fine.
                    # Let's keep them for now, the inter-entity part below will be more specific to the *group* of entities.
                    if relations:
                        rel_descs = []
                        for rel in relations:
                            target_name = rel.get("target_name", f"Entity {rel['target_id']}")
                            rel_descs.append(f"is '{rel['type']}' {target_name} (ID: {rel['target_id']})")
                        if rel_descs:
                            summary += " It " + ", and ".join(rel_descs) + "."
                    all_summaries.append(summary)
                else:
                    all_summaries.append(f"Entity ID '{entity_id_str}' not found or has no details.")
            except Exception as e:
                logger.error(f"Error querying Neo4j for individual entity ID '{entity_id_str}': {e}", exc_info=True)
                all_summaries.append(f"Could not retrieve context for entity ID '{entity_id_str}'.")

        # 2. Inter-Entity Relationships (if multiple unique entity IDs were provided)
        unique_entity_ids_list = list(processed_entity_ids)
        if len(unique_entity_ids_list) > 1:
            inter_entity_rels_summary: List[str] = []
            try:
                # Query for direct relationships between the provided entities
                query_inter_direct = """
                MATCH (e1:Entity)-[r]-(e2:Entity)
                WHERE e1.entity_id IN $entity_ids AND e2.entity_id IN $entity_ids AND elementId(e1) < elementId(e2) // Avoid duplicates and self-loops in listing
                RETURN e1.name AS entity1_name, e1.entity_id AS entity1_id,
                       type(r) AS relationship_type,
                       e2.name AS entity2_name, e2.entity_id AS entity2_id
                """
                result_direct = session.run(query_inter_direct, entity_ids=unique_entity_ids_list)
                direct_rels_found = False
                for record in result_direct:
                    if not direct_rels_found:
                        inter_entity_rels_summary.append("\nRelationships between these entities:")
                        direct_rels_found = True
                    e1_name = record.get("entity1_name", f"Entity {record['entity1_id']}")
                    e2_name = record.get("entity2_name", f"Entity {record['entity2_id']}")
                    inter_entity_rels_summary.append(f"- {e1_name} (ID: {record['entity1_id']}) is '{record['relationship_type']}' {e2_name} (ID: {record['entity2_id']}).")
                
                # Optional: Query for relationships via one intermediate node (paths of length 2)
                # This can be verbose, so use with caution or add more filters if needed.
                # For now, let's keep it commented or simple to avoid too much output.
                # query_inter_indirect = """
                # MATCH (e1:Entity)-[r1]-(mid:Entity)-[r2]-(e2:Entity)
                # WHERE e1.entity_id IN $entity_ids AND e2.entity_id IN $entity_ids 
                #   AND e1.entity_id <> e2.entity_id 
                #   AND NOT mid.entity_id IN $entity_ids // Intermediate node is not one of the main ones
                #   AND elementId(e1) < elementId(e2) // Avoid duplicate paths in reverse
                # RETURN e1.name AS e1_name, e1.entity_id AS e1_id, 
                #        type(r1) AS r1_type, 
                #        mid.name AS mid_name, mid.entity_id AS mid_id,
                #        type(r2) AS r2_type, 
                #        e2.name AS e2_name, e2.entity_id AS e2_id
                # LIMIT 5 // Limit indirect relationships to keep summary concise
                # """
                # result_indirect = session.run(query_inter_indirect, entity_ids=unique_entity_ids_list)
                # indirect_rels_found = False
                # for record in result_indirect:
                #     if not indirect_rels_found and not direct_rels_found: # Only add header if no direct rels summary either
                #          inter_entity_rels_summary.append("\nIndirect relationships between these entities:")
                #     elif not indirect_rels_found:
                #          inter_entity_rels_summary.append("\nAlso, some indirect relationships:") # If direct ones were already listed
                #     indirect_rels_found = True
                #     e1_name = record.get("e1_name", f"Entity {record['e1_id']}")
                #     mid_name = record.get("mid_name", f"Entity {record['mid_id']}")
                #     e2_name = record.get("e2_name", f"Entity {record['e2_id']}")
                #     inter_entity_rels_summary.append(f"- {e1_name} (ID: {record['e1_id']}) is '{record['r1_type']}' {mid_name} (ID: {record['mid_id']}), which is '{record['r2_type']}' {e2_name} (ID: {record['e2_id']}).")

                if inter_entity_rels_summary:
                    all_summaries.extend(inter_entity_rels_summary)
                elif len(unique_entity_ids_list) > 1: # only if we tried to find inter-entity rels
                    all_summaries.append("\nNo direct relationships were found between the specified group of entities.")

            except Exception as e:
                logger.error(f"Error querying Neo4j for inter-entity relationships between {unique_entity_ids_list}: {e}", exc_info=True)
                all_summaries.append("Could not retrieve inter-entity relationship context.")

    return (" ".join(all_summaries) if all_summaries else "No graph context found for the given entities.")


def build_prompt(query: str, content_items: List[Dict[str, Any]], graph_summary: str, audio_transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if USE_CAPTIONING:
        system_message = (
            "You are a helpful multimodal assistant. Your task is to analyze the user's query, "
            "any provided image captions, a summary of graph database context, and relevant audio transcripts "
            "to provide a comprehensive answer. Refer to image captions and audio snippets if they are relevant to the query."
        )
    else:
        system_message = (
            "You are a helpful multimodal assistant. Your task is to analyze the user's query, "
            "any provided images, a summary of graph database context, and relevant audio transcripts "
            "to provide a comprehensive answer. Refer to images and audio snippets if they are relevant to the query."
        )
    
    user_message_content: List[Dict[str, Any]] = []
    
    text_content = f"User Query: {query}\n\nGraph Context:\n{graph_summary}"
    
    if audio_transcripts:
        text_content += "\n\nRelevant Audio Snippets:"
        for i, audio_info in enumerate(audio_transcripts):
            text_content += f"\n  Snippet {i+1} (Speaker: {audio_info.get('speaker_label', 'Unknown')}, Time: {audio_info.get('start_time_seconds'):.2f}s - {audio_info.get('end_time_seconds'):.2f}s, Score: {audio_info.get('score'):.3f}): "
            text_content += f'"{audio_info.get("transcript_text")}"'
    else:
        text_content += "\n\nNo relevant audio snippets were found for this query."

    # Handle content based on mode
    if USE_CAPTIONING:
        # Add captions as text
        caption_texts = []
        for item in content_items:
            caption_text = item.get("caption_text")
            if caption_text:
                entity_info = f"{item.get('class_name', 'Unknown')} (ID: {item.get('entity_id', 'Unknown')}, Score: {item.get('score', 0):.3f})"
                caption_texts.append(f"- {entity_info}: {caption_text}")
        
        if caption_texts:
            text_content += "\n\nRelevant Image Captions:"
            for caption in caption_texts:
                text_content += f"\n  {caption}"
        else:
            text_content += "\n\nNo relevant image captions were found for this query."
    else:
        # Add images as before
        images_to_add = []
        for item in content_items:
            image_data = item.get("image_data")
            if image_data:
                images_to_add.append(image_data)
        
        if images_to_add:
            text_content += "\n\nRelevant Images are provided below:"
        else:
            text_content += "\n\nNo relevant images were found or provided for this query."
    
    user_message_content.append({"type": "text", "text": text_content})
    
    # Add images only if not in captioning mode
    if not USE_CAPTIONING:
        for item in content_items:
            image_data = item.get("image_data")
            if image_data:
                try:
                    # Convert PIL image to base64 data URI for the prompt
                    buffered = io.BytesIO()
                    image_data.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    base64_image_uri = f"data:image/png;base64,{img_str}"
                    user_message_content.append({
                            "type": "image_url",
                            "image_url": {"url": base64_image_uri, "detail": "auto"},
                        })
                except Exception as e:
                    logger.error(f"Error converting image to base64 for prompt: {e}", exc_info=True)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content},
    ]


def interactive_loop():
    if not openai_client:
        logger.critical("OpenAI client is not initialized. Interactive loop cannot start.")
        return

    logger.info("Starting interactive chat assistant. Type 'exit' or 'quit' to end.")
    while True:
        try:
            user_query = input("You: ").strip()
            if user_query.lower() in ["exit", "quit"]:
                logger.info("Exiting interactive loop.")
                break
            if not user_query:
                continue

            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir_name = timestamp_str
            save_path_base = os.path.join("runs", "track", "exp", "infer")
            query_save_path = os.path.join(save_path_base, save_dir_name)
            try:
                os.makedirs(query_save_path, exist_ok=True)
                logger.info(f"Created directory for current query: {query_save_path}")
            except Exception as e_mkdir:
                logger.error(f"Could not create directory {query_save_path}: {e_mkdir}")
            
            interaction_log_data = {
                "user_query": user_query,
                "retrieval_config": { # New logging for retrieval strategy
                    "target_entities": TARGET_ENTITY_TOP_K,
                    "target_frames": TARGET_FRAME_TOP_K,
                    "retrieval_multiplier": RETRIEVAL_MULTIPLIER,
                    "min_similarity_threshold": MIN_SIMILARITY_THRESHOLD,
                    "target_audio_segments": TARGET_AUDIO_TOP_K, # Log audio retrieval config
                    "audio_retrieval_multiplier": AUDIO_RETRIEVAL_MULTIPLIER,
                    "min_audio_similarity_threshold": MIN_AUDIO_SIMILARITY_THRESHOLD
                },
                "retrieved_graph_context": "",
                "retrieved_audio_transcripts": [], # For logging
                "llm_prompt_text": "",
                "saved_image_filenames": [],
                "llm_response": ""
            }

            logger.info(f"Embedding user query: '{user_query[:50]}...'")
            query_embedding = embed_text(user_query)

            if query_embedding is None:
                logger.error("Failed to embed user query. Skipping retrieval and OpenAI call.")
                print("Assistant: Sorry, I couldn't process your query due to an embedding issue.")
                interaction_log_data["llm_response"] = "Failed to embed query."
                # Log and continue to next iteration or handle error
                if query_save_path:
                    log_file_path = os.path.join(query_save_path, "interaction_log.json")
                    try:
                        with open(log_file_path, 'w') as f_log:
                            json.dump(interaction_log_data, f_log, indent=4)
                        logger.info(f"Interaction log saved (with error): {log_file_path}")
                    except Exception as e_save_log:
                        logger.error(f"Failed to save interaction log to {log_file_path}: {e_save_log}", exc_info=True)
                continue
            
            logger.info(f"Query embedded successfully, shape: {query_embedding.shape}")

            content_type = "captions" if USE_CAPTIONING else "images"
            logger.info(f"Retrieving similar {content_type} for query: '{user_query[:50]}...'")
            retrieved_content_infos = retrieve_similar_content(
                query_text=user_query, 
                query_embedding=query_embedding, 
                qdrant_client_instance=qdrant_client,
                logger_instance=logger
            )
            logger.info(f"Retrieved {len(retrieved_content_infos)} content infos from Qdrant.")

            entity_ids_for_graph_context: List[str] = []
            saved_image_filenames_for_log: List[str] = []

            if not retrieved_content_infos:
                logger.info(f"No {content_type} were retrieved from Qdrant.")
            else:
                logger.info(f"Processing {len(retrieved_content_infos)} retrieved content infos:")
                for i, info in enumerate(retrieved_content_infos):
                    content_type_item = info.get("type", "unknown_type")
                    content_id = info.get("vector_id", "unknown_id")
                    entity_id = info.get("entity_id", "N/A")
                    class_name = info.get("class_name", "N/A")
                    score = info.get("score", 0.0)
                    
                    if USE_CAPTIONING:
                        caption_text = info.get("caption_text", "")
                        log_message = f"  Info {i+1}: ID={content_id}, Type={content_type_item}, EntityID={entity_id}, Class={class_name}, Score={score:.4f}, Caption='{caption_text[:30]}{'...' if len(caption_text) > 30 else ''}'"
                    else:
                        log_message = f"  Info {i+1}: ID={content_id}, Type={content_type_item}, EntityID={entity_id}, Class={class_name}, Score={score:.4f}"
                        if info.get("image_data"):
                            log_message += " -> Has image data."
                            try:
                                image_filename = f"retrieved_image_{i}.png"
                                full_image_save_path = os.path.join(query_save_path, image_filename)
                                info["image_data"].save(full_image_save_path)
                                saved_image_filenames_for_log.append(image_filename)
                                logger.info(f"Saved image: {full_image_save_path}")
                            except Exception as e_save_img:
                                logger.error(f"Failed to save image {i} for query: {e_save_img}", exc_info=True)
                        else:
                            log_message += " -> No image data."
                    
                    logger.info(log_message)
                    if info.get("type") == "entity" and info.get("entity_id"):
                        entity_ids_for_graph_context.append(str(info["entity_id"]))
                        
                interaction_log_data["saved_image_filenames"] = saved_image_filenames_for_log

            logger.info(f"Total content items prepared for prompt: {len(retrieved_content_infos)}.")
            entity_ids_for_graph_context = list(set(entity_ids_for_graph_context))
            graph_summary_text = "No relevant entities found from image search for graph context."
            if entity_ids_for_graph_context:
                logger.info(f"Retrieving graph context for entity IDs: {entity_ids_for_graph_context}")
                graph_summary_text = retrieve_graph_context(entity_ids_for_graph_context)
            else:
                logger.info("No specific entity IDs from image search to query graph context.")
            interaction_log_data["retrieved_graph_context"] = graph_summary_text

            logger.info("Building prompt for OpenAI ChatCompletion.")
            # Retrieve relevant audio transcripts
            retrieved_audio_segments = retrieve_relevant_audio_transcripts(
                query_embedding=query_embedding,
                qdrant_client_instance=qdrant_client,
                logger_instance=logger
            )
            interaction_log_data["retrieved_audio_transcripts"] = retrieved_audio_segments # Log retrieved audio
            logger.info(f"Retrieved {len(retrieved_audio_segments)} audio segments for the prompt.")

            if not retrieved_audio_segments:
                logger.info("No audio segments were retrieved from Qdrant or selected for the prompt.")
            else:
                logger.info(f"Processing {len(retrieved_audio_segments)} retrieved audio segments for prompt:")
                for i, audio_info in enumerate(retrieved_audio_segments):
                    segment_id = audio_info.get("audio_segment_unique_id", "unknown_id")
                    speaker = audio_info.get("speaker_label", "N/A")
                    score = audio_info.get("score", 0.0)
                    text_preview = audio_info.get("transcript_text", "")[:50] # Preview of text
                    log_message = f"  Audio Segment {i+1}: ID={segment_id}, Speaker={speaker}, Score={score:.4f}, Text=\"{text_preview}...\" -> Added to prompt."
                    logger.info(log_message)

            chat_messages = build_prompt(user_query, retrieved_content_infos, graph_summary_text, retrieved_audio_segments)
            if chat_messages and len(chat_messages) > 1 and isinstance(chat_messages[1].get('content'), list):
                for content_item in chat_messages[1]['content']:
                    if content_item.get('type') == 'text':
                        interaction_log_data["llm_prompt_text"] = content_item['text']
                        break
            
            logger.info(f"Sending request to OpenAI model '{OPENAI_CHAT_MODEL}'.")
            completion_response = openai_client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=chat_messages, 
                max_tokens=1500,
            )
            assistant_reply = completion_response.choices[0].message.content
            print(f"Assistant: {assistant_reply}")
            interaction_log_data["llm_response"] = assistant_reply

        except KeyboardInterrupt:
            logger.info("\nUser interrupted (Ctrl+C). Exiting interactive loop.")
            break
        except openai.APIError as e:
            logger.error(f"OpenAI API Error: {e}", exc_info=True)
            print(f"Assistant: Sorry, I encountered an issue with the AI service: {getattr(e, 'message', str(e))}")
            interaction_log_data["llm_response"] = f"OpenAI API Error: {getattr(e, 'message', str(e))}"
        except Exception as e:
            logger.error(f"An unexpected error occurred in the interactive loop: {e}", exc_info=True)
            print("Assistant: Apologies, an unexpected error occurred. Please try again.")
            interaction_log_data["llm_response"] = f"Unexpected error: {str(e)}"
        finally:
            if query_save_path: 
                log_file_path = os.path.join(query_save_path, "interaction_log.json")
                try:
                    with open(log_file_path, 'w') as f_log:
                        json.dump(interaction_log_data, f_log, indent=4)
                    logger.info(f"Interaction log saved to: {log_file_path}")
                except Exception as e_save_log:
                    logger.error(f"Failed to save interaction log to {log_file_path}: {e_save_log}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive multimodal chat assistant using SentenceTransformer CLIP, Qdrant, Neo4j, and OpenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--neo4j-uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI. Can also be set via NEO4J_URI environment variable.",
    )
    parser.add_argument(
        "--neo4j-user", type=str, default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username. Can also be set via NEO4J_USER environment variable.",
    )
    parser.add_argument(
        "--neo4j-pass", type=str, dest="neo4j_password", default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password. Can also be set via NEO4J_PASSWORD environment variable.",
    )
    parser.add_argument(
        "--qdrant-host", type=str, default=os.environ.get("QDRANT_HOST", "localhost"),
        help="Qdrant host. Can also be set via QDRANT_HOST environment variable.",
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=int(os.environ.get("QDRANT_PORT", 6333)),
        help="Qdrant port. Can also be set via QDRANT_PORT environment variable.",
    )
    args = parser.parse_args()

    try:
        logger.info("Initializing components...")
        initialize_embedding_model()
        initialize_openai_client()
        initialize_qdrant_client(host=args.qdrant_host, port=args.qdrant_port)
        initialize_neo4j_driver(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
        initialize_minio_client_infer() # Initialize MinIO client for inference
        logger.info("All components initialized successfully for interactive mode.")
        interactive_loop()
    except ValueError as ve:
        logger.critical(f"Configuration error: {ve}", exc_info=True)
    except Exception as e:
        logger.critical(f"Application failed to start or encountered a critical error: {e}", exc_info=True)
    finally:
        if neo4j_driver:
            try:
                neo4j_driver.close()
                logger.info("Neo4j connection closed.")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)
        logger.info("Application shutdown.")


if __name__ == "__main__":
    main()
