# USE_CAPTIONING Feature

This document explains how to use the new `USE_CAPTIONING` feature that allows you to choose between storing images in MinIO or using text captioning for image content.

## Overview

The system now supports two modes:

1. **MinIO Storage Mode** (default): Images are stored in MinIO object storage and retrieved during inference
2. **Captioning Mode**: Images are converted to text captions and stored as metadata, reducing storage requirements

## Configuration

### Environment Variable

Set the `USE_CAPTIONING` environment variable:

```bash
export USE_CAPTIONING=True  # Enable captioning mode
export USE_CAPTIONING=False # Use MinIO storage (default)
```

### Command Line Flag

Use the `--use-captioning` flag when running `track.py`:

```bash
python tracking/track.py --source video.mp4 --use-captioning
```

## How It Works

### Captioning Mode (USE_CAPTIONING=True)

1. **Frame Processing**: When a new frame needs to be stored, the system:
   - Creates visual embeddings for similarity checking (using FRAME_SIMILARITY_THRESHOLD)
   - Generates a text caption using the moondream model
   - Stores the caption as metadata instead of the image

2. **Entity Processing**: When a new entity crop needs to be stored:
   - Creates visual embeddings for similarity checking (using ENTITY_SIMILARITY_THRESHOLD)
   - Generates a text caption for the entity crop
   - Stores the caption as metadata instead of the image

3. **Inference**: During query processing:
   - Retrieves relevant captions based on similarity search
   - Includes captions in the LLM prompt as text context
   - No images are sent to the LLM

### MinIO Storage Mode (USE_CAPTIONING=False)

1. **Frame/Entity Processing**: Images are stored in MinIO object storage
2. **Inference**: Images are retrieved from MinIO and sent to the LLM

## Benefits

### Captioning Mode
- **Reduced Storage**: Text captions require much less storage than images
- **Privacy**: No actual images are stored, only descriptive text
- **Faster Inference**: No need to download and process images during queries
- **Better Text Search**: Captions can be more easily searched and matched

### MinIO Mode
- **Visual Context**: Full visual information is preserved
- **Multimodal LLM Support**: Images can be directly processed by vision-capable models
- **Higher Fidelity**: No information loss from caption generation

## Configuration Examples

### Using Environment Variable

```bash
# Enable captioning mode
export USE_CAPTIONING=True
python tracking/track.py --source video.mp4 --save-dataset

# Use MinIO storage
export USE_CAPTIONING=False
python tracking/track.py --source video.mp4 --save-dataset
```

### Using Command Line Flag

```bash
# Enable captioning mode
python tracking/track.py --source video.mp4 --save-dataset --use-captioning

# Use MinIO storage (default)
python tracking/track.py --source video.mp4 --save-dataset
```

### For Inference

The inference script (`tracking/infer.py`) automatically detects the mode based on the `USE_CAPTIONING` environment variable:

```bash
# For captioning mode
export USE_CAPTIONING=True
python tracking/infer.py

# For MinIO mode
export USE_CAPTIONING=False
python tracking/infer.py
```

## Technical Details

### Storage Schema

**Captioning Mode:**
- Entities: `image_caption` field in payload
- Frames: `image_caption` field in payload
- No MinIO storage

**MinIO Mode:**
- Entities: `image_minio_id` field in payload
- Frames: `minio_image_path` field in payload
- Images stored in MinIO bucket

### Similarity Thresholds

Both modes use the same similarity thresholds for embedding comparison:
- `FRAME_SIMILARITY_THRESHOLD` (default: 0.8)
- `ENTITY_SIMILARITY_THRESHOLD` (default: 0.8)

These control when to reuse existing embeddings vs creating new ones.

## Troubleshooting

### Caption Model Issues

If the caption model fails to load:
```
[vectorize.py] Warning: USE_CAPTIONING=True but failed to import caption module
[vectorize.py] Falling back to MinIO storage mode
```

Ensure the caption model dependencies are installed and the model is accessible.

### Mixed Mode Data

If you have data stored in one mode and switch to another, the system will handle it gracefully:
- Missing image data will be logged as warnings
- Missing captions will be logged as warnings
- Inference will work with available data

### Storage Switching

To switch modes on existing data:
1. Set the new `USE_CAPTIONING` value
2. Use `--clear-prev-runs` to start fresh, or
3. Accept that inference will use whatever data format exists 