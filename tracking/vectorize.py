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
from typing import Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Connect to Qdrant running in Docker
qdrant = QdrantClient(host='localhost', port=6333)
COLLECTION_NAME = 'entities'
FRAME_COLLECTION_NAME = 'frames'

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
# elif MODEL_TYPE == "siglip": # Example if you want to support SigLIP via env var too
#     from transformers import AutoModel, AutoProcessor
#     try:
#         embedding_model = AutoModel.from_pretrained(MODEL_NAME)
#         processor = AutoProcessor.from_pretrained(MODEL_NAME)
#         embedding_model.eval()
#         print(f"[vectorize.py] Successfully loaded SigLIP model: {MODEL_NAME}")
#     except Exception as e:
#         print(f"[vectorize.py] Error loading SigLIP model {MODEL_NAME}: {e}.")
#         raise
else:
    raise ValueError(f"[vectorize.py] Unsupported MODEL_TYPE '{MODEL_TYPE}' configured for {MODEL_NAME}")

def reinitialize_collections():
    """Deletes and recreates the Qdrant collections with the current EMBED_DIM."""
    print(f"[vectorize.py] Reinitializing Qdrant collections with dimension: {EMBED_DIM}")
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
        print("[vectorize.py] Successfully reinitialized Qdrant collections.")
    except Exception as e:
        print(f"[vectorize.py] Error reinitializing Qdrant collections: {e}")
        raise

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
    """Add a new entity to Qdrant with embedding, metadata, and image crop (as base64). Returns vector_id."""
    img_b64 = None # Initialize
    if crop is not None:
        try:
            pil_img = Image.fromarray(crop[..., ::-1]) if crop.ndim == 3 and crop.shape[2] == 3 else Image.fromarray(crop)
            
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding entity crop to base64 for entity_id {entity_id}: {e}")

    payload = metadata.copy()
    if img_b64: # Only add image to payload if successfully created
        payload['image'] = img_b64
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
    """Add a new frame embedding to Qdrant. Optionally includes the frame image as base64. Returns the vector_id."""
    payload = {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
    }
    if additional_metadata:
        payload.update(additional_metadata)
    
    img_b64 = None # Initialize img_b64
    if image is not None:
        try:
            pil_img = Image.fromarray(image[..., ::-1]) if image.ndim == 3 and image.shape[2] == 3 else Image.fromarray(image)
            
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding frame image to base64 for frame_idx {frame_idx}: {e}")

    if img_b64: # Add to payload only if successfully created
        payload['image'] = img_b64

    vector_id = str(uuid.uuid4())
    point = PointStruct(
        id=vector_id,
        vector=embedding.tolist(),
        payload=payload
    )
    qdrant.upsert(collection_name=FRAME_COLLECTION_NAME, points=[point])
    return vector_id

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