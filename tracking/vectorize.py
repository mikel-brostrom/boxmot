import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import io
import base64
import uuid
import os
import traceback
from minio import Minio
from minio.error import S3Error
from minio.deleteobjects import DeleteObject
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Connect to Qdrant running in Docker
qdrant = QdrantClient(host='localhost', port=6333)
COLLECTION_NAME = 'entities'
FRAME_COLLECTION_NAME = 'frames'
AUDIO_COLLECTION_NAME = 'audio_transcripts'

# --- MinIO Configuration ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "boxmot-images")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "False").lower() == "true"

minio_client = None
try:
    print(f"[vectorize.py] Initializing MinIO client for endpoint: {MINIO_ENDPOINT}, bucket: {MINIO_BUCKET_NAME}")
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL
    )
    # Check if the bucket exists, create if not
    found = minio_client.bucket_exists(MINIO_BUCKET_NAME)
    if not found:
        minio_client.make_bucket(MINIO_BUCKET_NAME)
        print(f"[vectorize.py] MinIO bucket '{MINIO_BUCKET_NAME}' created.")
    else:
        print(f"[vectorize.py] MinIO bucket '{MINIO_BUCKET_NAME}' already exists.")
    print(f"[vectorize.py] MinIO client initialized successfully.")
except Exception as e:
    print(f"[vectorize.py] Error initializing MinIO client: {e}. Please ensure MinIO is running and configured.")
    minio_client = None # Ensure client is None if initialization fails

def clear_minio_bucket():
    """Clears all objects from the configured MinIO bucket."""
    if not minio_client:
        print("[vectorize.py] MinIO client not initialized. Cannot clear bucket.")
        return

    try:
        print(f"[vectorize.py] Listing objects in MinIO bucket: {MINIO_BUCKET_NAME} for deletion...")
        objects_to_delete_iterable = minio_client.list_objects(MINIO_BUCKET_NAME, recursive=True)
        
        delete_object_payload = []
        for obj_data in objects_to_delete_iterable:
            if obj_data and obj_data.object_name:
                delete_object_payload.append(DeleteObject(obj_data.object_name))
            elif obj_data:
                print(f"[vectorize.py] Skipping object without a name: {obj_data.etag if obj_data.etag else 'Unknown ETag'}")

        if not delete_object_payload:
            print(f"[vectorize.py] MinIO bucket '{MINIO_BUCKET_NAME}' is already empty or contains no deletable named objects.")
            return

        print(f"[vectorize.py] Prepared {len(delete_object_payload)} DeleteObject instances for deletion from MinIO bucket '{MINIO_BUCKET_NAME}'.")
        errors_iterator = minio_client.remove_objects(MINIO_BUCKET_NAME, delete_object_payload)
        
        error_count = 0
        for error_detail in errors_iterator:
            if hasattr(error_detail, 'object_name') and hasattr(error_detail, 'message'):
                object_name = getattr(error_detail, 'object_name', 'UnknownObject')
                bucket_name = getattr(error_detail, 'bucket_name', MINIO_BUCKET_NAME)
                message = getattr(error_detail, 'message', 'No message')
                code = getattr(error_detail, 'code', 'N/A')
                print(f"[vectorize.py] Error deleting object {object_name} from MinIO bucket {bucket_name}: {message} (Code: {code})")
            else:
                print(f"[vectorize.py] Unknown error type or malformed error during MinIO object deletion: {error_detail}")
            error_count += 1
        
        if error_count == 0:
            print(f"[vectorize.py] Successfully cleared all {len(delete_object_payload)} targeted objects from MinIO bucket '{MINIO_BUCKET_NAME}'.")
        else:
            print(f"[vectorize.py] Finished attempting to clear MinIO bucket '{MINIO_BUCKET_NAME}' with {error_count} errors for {len(delete_object_payload)} targeted objects.")

    except S3Error as exc:
        print(f"[vectorize.py] S3Error while clearing MinIO bucket: {exc}")
    except Exception as e:
        print(f"[vectorize.py] Unexpected error while clearing MinIO bucket: {e}")
        print(f"[vectorize.py] Traceback: {traceback.format_exc()}")

# --- Model Configuration based on Environment Variable ---
SUPPORTED_EMBEDDING_MODELS = {
    "clip-ViT-B-32": {"dimension": 512, "type": "sentence_transformer"},
    "clip-ViT-L-14": {"dimension": 768, "type": "sentence_transformer"}
    # Add other models here if needed, e.g., SigLIP if you want to switch back
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
EMBED_DIM = MODEL_CONFIG["dimension"]
MODEL_TYPE = MODEL_CONFIG["type"]

print(f"[vectorize.py] Using embedding model: {MODEL_NAME} (Dimension: {EMBED_DIM}, Type: {MODEL_TYPE})")

# Load the specified model
# Global model variable
embedding_model = None
# processor = None # SigLIP processor, no longer needed for SentenceTransformer

if MODEL_TYPE == "sentence_transformer":
    try:
        embedding_model = SentenceTransformer(MODEL_NAME, device='cpu') # Adjust device as needed e.g. 'cuda'
        embedding_model.eval() # Set to eval mode
        print(f"[vectorize.py] Successfully loaded SentenceTransformer model: {MODEL_NAME}")
    except Exception as e:
        print(f"[vectorize.py] Error loading SentenceTransformer model {MODEL_NAME}: {e}. Please ensure it's installed or a valid model name.")
        raise
else:
    raise ValueError(f"[vectorize.py] Unsupported MODEL_TYPE '{MODEL_TYPE}' configured for {MODEL_NAME}")

def reinitialize_collections():
    """Deletes and recreates the Qdrant collections with the current EMBED_DIM and clears MinIO bucket."""
    print(f"[vectorize.py] Reinitializing Qdrant collections with dimension: {EMBED_DIM} and clearing MinIO bucket.")
    
    # Clear MinIO bucket first
    clear_minio_bucket()
    
    try:
        collections_response = qdrant.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if COLLECTION_NAME in collection_names:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
        
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )

        if FRAME_COLLECTION_NAME in collection_names:
            qdrant.delete_collection(collection_name=FRAME_COLLECTION_NAME)

        qdrant.recreate_collection(
            collection_name=FRAME_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )

        if AUDIO_COLLECTION_NAME in collection_names:
            qdrant.delete_collection(collection_name=AUDIO_COLLECTION_NAME)
        qdrant.recreate_collection(
            collection_name=AUDIO_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
        print("[vectorize.py] Successfully reinitialized Qdrant collections (entities, frames, audio_transcripts).")
    except Exception as e:
        print(f"[vectorize.py] Error reinitializing Qdrant collections: {e}")
        raise

def get_qdrant_client():
    """Returns the initialized Qdrant client instance."""
    return qdrant

# Initial creation logic - make sure EMBED_DIM is used here too
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    print(f"[vectorize.py] Collection '{COLLECTION_NAME}' not found. Creating with dimension: {EMBED_DIM}")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )

if FRAME_COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    print(f"[vectorize.py] Collection '{FRAME_COLLECTION_NAME}' not found. Creating with dimension: {EMBED_DIM}")
    qdrant.recreate_collection(
        collection_name=FRAME_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )

# Removed old SigLIP model loading here as it's done based on MODEL_TYPE now

def embed_image(image: np.ndarray) -> np.ndarray:
    """Embed an image using the loaded SentenceTransformer model."""
    if embedding_model is None:
        raise RuntimeError("[vectorize.py] Embedding model is not loaded.")
    if MODEL_TYPE != "sentence_transformer":
        raise NotImplementedError(f"[vectorize.py] embed_image not implemented for MODEL_TYPE {MODEL_TYPE}")
    
    pil_img = Image.fromarray(image[..., ::-1]) if image.ndim == 3 and image.shape[2] == 3 else Image.fromarray(image)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    # SentenceTransformer.encode() can take a PIL image directly
    emb = embedding_model.encode(pil_img) 
    return emb.astype(np.float32).flatten() # Ensure it's flattened and float32

def embed_text(text: str) -> np.ndarray:
    """Embed text using the loaded SentenceTransformer model."""
    if embedding_model is None:
        raise RuntimeError("[vectorize.py] Embedding model is not loaded.")
    if MODEL_TYPE != "sentence_transformer":
        # Or if the model is image-only and doesn't support text
        raise NotImplementedError(f"[vectorize.py] embed_text not implemented or model {MODEL_NAME} doesn't support text for MODEL_TYPE {MODEL_TYPE}")

    # SentenceTransformer.encode() can take a string or list of strings
    emb = embedding_model.encode(text) 
    return emb.astype(np.float32).flatten() # Ensure it's flattened and float32

def embed_crop(image: np.ndarray) -> np.ndarray:
    """Embed a crop using the loaded SentenceTransformer model."""
    return embed_image(image) # Same logic for now

def search_entity(embedding: np.ndarray, threshold: float = 0.95):
    """Search for a matching entity in Qdrant. Returns (entity_id, vector_id, payload, score) if found above threshold."""
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding.tolist(),
        limit=1,
        with_payload=True
    )
    if hits and hits[0].score >= threshold:
        payload = hits[0].payload
        entity_id = payload.get('entity_id')
        vector_id = hits[0].id
        return entity_id, vector_id, payload, hits[0].score
    return None, None, None, None

def add_entity(embedding: np.ndarray, crop: np.ndarray, metadata: dict, entity_id=None):
    """Add a new entity to Qdrant with embedding, metadata, and stores its image crop in MinIO. Returns vector_id."""
    minio_object_name = None
    if crop is not None:
        try:
            pil_img = Image.fromarray(crop[..., ::-1]) if crop.ndim == 3 and crop.shape[2] == 3 else Image.fromarray(crop)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            image_bytes = buf.getvalue()
            minio_object_name = upload_image_to_minio(image_bytes, object_name_prefix=f"entity_crop_{entity_id or 'unk'}")
        except Exception as e:
            print(f"Error processing or uploading entity crop to MinIO for entity_id {entity_id}: {e}")

    payload = metadata.copy()
    if minio_object_name: # Only add minio id if upload was successful
        payload['image_minio_id'] = minio_object_name
    else:
        payload.pop('image', None) # Remove old base64 image field if it existed
        payload.pop('image_minio_id', None) # Ensure it's not there if upload failed
        
    payload['entity_id'] = entity_id # Ensure entity_id is in payload
    vector_id = str(uuid.uuid4())
    point = PointStruct(
        id=vector_id,
        vector=embedding.tolist(),
        payload=payload
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return vector_id

def add_frame_embedding(embedding: np.ndarray, frame_idx: int, timestamp: float, image: np.ndarray = None, additional_metadata: dict = None):
    """Add a new frame embedding to Qdrant. Optionally uploads the frame image to MinIO. Returns the vector_id."""
    payload = {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
    }
    if additional_metadata:
        payload.update(additional_metadata)
    
    minio_object_name = None
    if image is not None:
        try:
            pil_img = Image.fromarray(image[..., ::-1]) if image.ndim == 3 and image.shape[2] == 3 else Image.fromarray(image)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            image_bytes = buf.getvalue()
            minio_object_name = upload_image_to_minio(image_bytes, object_name_prefix=f"frame_{frame_idx}")
        except Exception as e:
            print(f"Error processing or uploading frame image to MinIO for frame_idx {frame_idx}: {e}")

    if minio_object_name: # Add to payload only if successfully uploaded
        payload['image_minio_id'] = minio_object_name
    else:
        payload.pop('image', None) # Remove old base64 image field if it existed
        payload.pop('image_minio_id', None) # Ensure it's not there if upload failed

    vector_id = str(uuid.uuid4())
    point = PointStruct(
        id=vector_id,
        vector=embedding.tolist(),
        payload=payload
    )
    qdrant.upsert(collection_name=FRAME_COLLECTION_NAME, points=[point])
    return vector_id

def add_audio_transcript_embedding(
    embedding: np.ndarray, 
    audio_segment_unique_id: str, 
    speaker_label: str, 
    start_time_seconds: float, 
    end_time_seconds: float, 
    transcript_text: str
):
    """Add a new audio transcript embedding and its metadata to Qdrant."""
    payload = {
        "audio_segment_unique_id": audio_segment_unique_id, # For linking back if needed, also the ID
        "speaker_label": speaker_label,
        "start_time_seconds": start_time_seconds,
        "end_time_seconds": end_time_seconds,
        "transcript_text": transcript_text,
        # overlapping_frame_ids and overlapping_entity_ids will be added by the linker later
    }
    
    # Use audio_segment_unique_id as the Qdrant point ID for direct lookup
    point = PointStruct(
        id=audio_segment_unique_id, 
        vector=embedding.tolist(),
        payload=payload
    )
    try:
        qdrant.upsert(collection_name=AUDIO_COLLECTION_NAME, points=[point])
        print(f"[vectorize.py] Successfully added audio transcript embedding for ID: {audio_segment_unique_id}")
        return audio_segment_unique_id # Return the ID used for the point
    except Exception as e:
        print(f"[vectorize.py] Error adding audio transcript embedding for ID {audio_segment_unique_id}: {e}")
        return None

def get_entity_metadata(vector_id):
    """Retrieve metadata for a given vector id."""
    res = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[vector_id], with_payload=True)
    if res:
        return res[0].payload
    return None

def calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if emb1 is None or emb2 is None:
        return 0.0
    # Ensure embeddings are 1D arrays (vectors)
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    
    dot_product = np.dot(emb1, emb2)
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    
    if norm_emb1 == 0 or norm_emb2 == 0:
        return 0.0  # Avoid division by zero if a vector is all zeros
        
    similarity = dot_product / (norm_emb1 * norm_emb2)
    return float(similarity) 

def upload_image_to_minio(image_bytes: bytes, object_name_prefix: str = "image") -> Optional[str]:
    """Uploads image bytes to MinIO and returns the object name."""
    if not minio_client:
        print("[vectorize.py] MinIO client not initialized. Cannot upload image.")
        return None
    try:
        # Generate a unique object name
        object_name = f"{object_name_prefix}_{uuid.uuid4()}.png"
        image_stream = io.BytesIO(image_bytes)
        minio_client.put_object(
            MINIO_BUCKET_NAME,
            object_name,
            image_stream,
            length=len(image_bytes),
            content_type='image/png'
        )
        print(f"[vectorize.py] Successfully uploaded {object_name} to MinIO bucket {MINIO_BUCKET_NAME}.")
        return object_name
    except S3Error as exc:
        print(f"[vectorize.py] Error uploading to MinIO: {exc}")
        return None 