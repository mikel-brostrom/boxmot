# Tracking Module Documentation

This directory contains the core logic for the multimodal analysis pipeline, including video and audio processing, object tracking, data vectorization, data linking, graph database interaction, and multimodal inference.

## Overall Pipeline

The system is orchestrated by `track.py` for batch processing and `infer.py` for interactive querying.

**Batch Processing Pipeline (Orchestrated by `track.py`):**

1.  **Input:** A video file.
2.  **Video Processing (`video_processing.py`):**
    *   Detects objects (e.g., using YOLO).
    *   Tracks objects across frames (e.g., using DeepOCSORT).
    *   Generates visual embeddings for frames and detected entity crops using `vectorize.py`.
    *   Stores frame images and entity crops in MinIO (via `vectorize.py`).
    *   Stores visual embeddings (frames, entities) in Qdrant (via `vectorize.py`).
    *   Calculates spatial relationships between entities within a frame (`relate.py`).
    *   Outputs an `initial_dataset.json` with frame-by-frame visual data, Qdrant vector IDs, MinIO object IDs for images, and spatial relationships.
3.  **Audio Processing (`audio_processing.py`):**
    *   Extracts audio from the video.
    *   Performs speaker diarization (e.g., using Pyannote).
    *   Transcribes audio segments for each speaker (e.g., using NVIDIA NeMo Parakeet).
    *   Generates embeddings for audio transcripts using `vectorize.py`.
    *   Stores audio transcript embeddings in Qdrant (via `vectorize.py`).
    *   Outputs an `*_audio_segments.json` file with diarized and transcribed audio data, including Qdrant vector IDs for transcripts.
4.  **Cross-Modal Linking (`link_data.py`):**
    *   Consumes `initial_dataset.json` (from video processing) and `*_audio_segments.json` (from audio processing).
    *   Links audio segments to video frames based on temporal overlap.
    *   Updates Qdrant payloads:
        *   `frames` collection: Adds `overlapping_audio_segment_ids` to frame payloads.
        *   `audio_transcripts` collection: Adds `overlapping_frame_ids` and `overlapping_entity_ids` to audio transcript payloads.
    *   Outputs an `enriched_dataset.json` which augments the video data with audio linkage information.
5.  **Graph Population (`graphify.py`):**
    *   Consumes the `enriched_dataset.json`.
    *   Populates a Neo4j graph database with nodes for frames and entities, and relationships between them (temporal, detection, spatial, etc.).

**Interactive Inference Pipeline (`infer.py`):**

1.  **User Input:** A natural language text query.
2.  **Query Embedding:** The text query is embedded using `vectorize.py`.
3.  **Multimodal Retrieval:**
    *   **Qdrant Search:** Searches Qdrant for relevant visual vectors (frames, entities) and audio transcript vectors based on the query embedding.
    *   **MinIO Retrieval:** Retrieves corresponding frame/entity images from MinIO using object IDs found in Qdrant payloads.
    *   **Neo4j Context:** Retrieves graph context (entity relationships, frame appearances) for entities identified in the retrieved visual data.
4.  **Prompt Construction:** Builds a multimodal prompt for an LLM (e.g., OpenAI GPT-4 Vision) containing:
    *   Original user query.
    *   Retrieved images (from MinIO).
    *   Retrieved audio transcript snippets (from Qdrant).
    *   Retrieved graph context summary (from Neo4j).
5.  **LLM Interaction:** Sends the prompt to the LLM.
6.  **Response:** Presents the LLM's answer to the user.

## Files

### `track.py`

The main orchestrator for the batch processing pipeline. It coordinates `video_processing.py`, `audio_processing.py`, `link_data.py`, and `graphify.py`.

**Key Functionalities:**
- Parses command-line arguments for all stages.
- Optionally clears previous run data and reinitializes Qdrant collections and MinIO bucket via `vectorize.py`.
- Calls `process_video()` to handle video analysis, embedding, and initial data generation.
- Calls `process_audio()` to handle audio analysis, transcription, and embedding.
- Calls `link_multimodal_data()` to link video and audio data and update Qdrant.
- Calls `run_graphification()` to populate the Neo4j graph.
- Aggregates and saves metrics from all stages.

**Command-line Arguments:**
- Includes arguments for video processing (YOLO models, tracking methods, etc.).
- Includes arguments for audio processing (`--hf-token`).
- Includes arguments for Neo4j connection (`--neo4j-uri`, `--neo4j-user`, `--neo4j-password`).
- Orchestrator controls like `--clear-prev-runs`.
- Output controls like `--project`, `--name`.

### `video_processing.py`

Handles object detection, tracking, ReID, visual embedding generation, and saving video-related outputs.

**Key Functionalities:**
- Uses YOLO for object detection and a specified tracking algorithm.
- Processes video frame by frame.
- Embeds frame images and detected entity crops via `vectorize.embed_image()` and `vectorize.embed_crop()`.
- Stores frame images and entity crops in MinIO via `vectorize.upload_image_to_minio()`.
- Stores visual embeddings in Qdrant `frames` and `entities` collections via `vectorize.add_frame_embedding()` and `vectorize.add_entity()`.
- Uses `relate.py` to compute spatial relationships between entities in a frame.
- Outputs an `initial_dataset.json` containing per-frame data (entities, Qdrant vector IDs, MinIO object IDs, relationships).
- Optionally saves annotated video and entity crops (delegated to Ultralytics).

### `audio_processing.py`

Manages the audio analysis pipeline.

**Key Functionalities:**
- Extracts audio from the input video using `ffmpeg`.
- Performs speaker diarization using `pyannote.audio`.
- Transcribes diarized audio segments using an ASR model (e.g., NVIDIA NeMo Parakeet).
- Embeds audio transcripts via `vectorize.embed_text()`.
- Stores transcript embeddings in the Qdrant `audio_transcripts` collection via `vectorize.add_audio_transcript_embedding()`.
- Outputs an `*_audio_segments.json` file with detailed segment information.

### `link_data.py`

Responsible for linking data from video and audio processing stages.

**Key Functionalities:**
- Takes `initial_dataset.json` (video) and `*_audio_segments.json` (audio) as input.
- Identifies temporal overlaps between video frames and audio segments.
- Augments the video data by adding `last_audio_segment_id` and `overlapping_audio_segment_ids` to each frame's information.
- Updates Qdrant payloads:
    - `frames` collection: Adds `overlapping_audio_segment_ids`.
    - `audio_transcripts` collection: Adds `overlapping_frame_ids` and `overlapping_entity_ids`.
- Outputs an `enriched_dataset.json` file.

### `vectorize.py`

A utility module for creating and managing vector embeddings and interacting with Qdrant and MinIO.

**Key Functionalities:**
- Initializes embedding models (e.g., SentenceTransformer CLIP for vision and text).
- Initializes Qdrant and MinIO clients.
- `reinitialize_collections()`: Clears the MinIO bucket and deletes/recreates Qdrant collections (`entities`, `frames`, `audio_transcripts`) with appropriate dimensions.
- Embedding functions: `embed_image()`, `embed_crop()`, `embed_text()`.
- Qdrant interaction:
    - `add_entity()`: Adds entity crop embedding to `entities` collection.
    - `add_frame_embedding()`: Adds frame image embedding to `frames` collection.
    - `add_audio_transcript_embedding()`: Adds audio transcript embedding to `audio_transcripts` collection.
    - `search_entity()`: Searches for similar entities.
    - `get_entity_metadata()`: Retrieves entity metadata.
- MinIO interaction:
    - `upload_image_to_minio()`: Uploads image bytes (frames, crops) to MinIO.
    - `clear_minio_bucket()`: Deletes all objects from the MinIO bucket.
- `calculate_cosine_similarity()`.

**Environment Variables Used:**
- `EMBEDDING_MODEL_NAME`: Specifies the SentenceTransformer model.
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`, `MINIO_USE_SSL`: MinIO connection details.
- Qdrant host/port are typically passed as arguments or use defaults.

**Qdrant Collections:**
- `entities`: Stores embeddings of detected object crops. Payload includes `entity_id`, `class_name`, `confidence`, `image_minio_id`.
- `frames`: Stores embeddings of video frames. Payload includes `frame_idx`, `timestamp`, `image_minio_id`.
- `audio_transcripts`: Stores embeddings of audio transcripts. Payload includes `audio_segment_unique_id`, `speaker_label`, `start_time_seconds`, `end_time_seconds`, `transcript_text`. Payloads are updated by `link_data.py` with `overlapping_frame_ids` and `overlapping_entity_ids`.

### `graphify.py`

Populates a Neo4j graph database from the `enriched_dataset.json`.

**Key Functionalities:**
- Connects to Neo4j.
- `clear_graph()`: Optionally clears existing graph data.
- `create_constraints()`: Ensures uniqueness for `Frame.frame_idx` and `Entity.entity_id`.
- `graphify_dataset()`:
    - Iterates through `enriched_dataset.json`.
    - Creates/updates `Frame` nodes (properties: `frame_idx`, `timestamp`, `geo`, `latest_vector_id` (Qdrant ID), `name`, `last_audio_segment_id`, `overlapping_audio_segment_ids`).
    - Creates `:NEXT`/`:PREV` relationships between `Frame` nodes.
    - Creates/updates `Entity` nodes (properties: `entity_id`, `class_name`, `latest_vector_id` (Qdrant ID), `name`).
    - Creates `:DETECTED_IN` relationships from `Entity` to `Frame` (properties: `confidence`, `vector_id` (Qdrant ID of the specific detection instance)).
    - Creates inter-entity spatial relationships (e.g., `:near`, `:left_of`).

**Environment Variables Used:**
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection details.

**Neo4j Graph Schema (Key Aspects):**
- **Nodes:** `Frame`, `Entity`.
- **Relationships:** `:NEXT`, `:PREV`, `:DETECTED_IN`, spatial relationships (e.g., `:near`).
- `Frame` nodes now store audio linkage information.

### `infer.py`

Interactive command-line interface for multimodal querying.

**Key Functionalities:**
- Initializes SentenceTransformer, Qdrant, Neo4j, MinIO, and OpenAI clients.
- `interactive_loop()`:
    - Embeds user's text query.
    - `retrieve_similar_images()`: Searches Qdrant (`entities`, `frames`), retrieves MinIO object IDs from payloads, and then downloads images from MinIO using `download_image_from_minio()`.
    - `retrieve_relevant_audio_transcripts()`: Searches Qdrant `audio_transcripts` collection.
    - `retrieve_graph_context()`: Gets context from Neo4j for entities identified from image search.
    - `build_prompt()`: Constructs a multimodal prompt with text query, graph summary, retrieved audio transcripts, and retrieved PIL images (from MinIO).
    - Sends prompt to OpenAI model and prints the response.
    - Logs interaction details.

**Environment Variables Used:**
- `OPENAI_MODEL`, `OPENAI_API_KEY`.
- `EMBEDDING_MODEL_NAME`.
- Neo4j, Qdrant, and MinIO connection variables (as above).

### `relate.py`

Computes basic spatial relationships between detected entities within a single frame. (Content largely unchanged, but its output is used by `video_processing.py` and subsequently `graphify.py`).

### `.env.example`

Template for environment variables. **Crucially, add MinIO and HF_TOKEN variables.**

```
# OpenAI
OPENAI_MODEL=gpt-4-vision-preview
OPENAI_API_KEY=your_openai_api_key_here

# Embedding Model
EMBEDDING_MODEL_NAME=clip-ViT-B-32 # Or other supported SentenceTransformer

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password # Change this for production

# Qdrant (host/port often configured via CLI args or defaults in code)
# QDRANT_HOST=localhost
# QDRANT_PORT=6333

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=boxmot-images # Must match docker-compose.yml
MINIO_USE_SSL=False # Set to True if MinIO is configured with SSL

# Hugging Face (for audio models like pyannote)
HF_TOKEN=your_huggingface_read_token_here
```

## Setup & Dependencies

- Python environment with packages from `pyproject.toml`.
- **Docker Services:** Qdrant, Neo4j, MinIO. Use the `docker-compose.yml` in the project root.
    ```bash
    docker-compose up -d qdrant neo4j minio
    ```
- **Environment Variables:** Create a `.env` file in the `tracking` directory (or project root) based on `.env.example` and fill in your credentials.

## Running the System

1.  **Start Docker Services:** (as above)
    Ensure all services are healthy. Check MinIO console at `http://localhost:9001`.

2.  **Run Batch Processing Pipeline (`track.py`):**
    ```bash
    python tracking/track.py --source <your_video_source> --save-dataset --metrics --clear-prev-runs --name my_experiment [--hf-token YOUR_HF_TOKEN_IF_NOT_IN_ENV]
    ```
    This will:
    *   Process video (`video_processing.py`) -> `initial_dataset.json`, Qdrant visual data, MinIO images.
    *   Process audio (`audio_processing.py`) -> `*_audio_segments.json`, Qdrant audio data.
    *   Link data (`link_data.py`) -> `enriched_dataset.json`, updated Qdrant payloads.
    *   Graphify data (`graphify.py`) -> Populates Neo4j.

3.  **Run Interactive Inference (`infer.py`):**
    ```bash
    python tracking/infer.py [--neo4j-uri ...] [--qdrant-host ...] # etc. if not using .env or defaults
    ```
    Type your questions at the prompt.
