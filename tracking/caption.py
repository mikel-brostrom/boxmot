import torch
import logging
import os
from PIL import Image
from transformers import AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

# --- Image Captioning Model Setup ---
CAPTIONING_MODEL = None
# Using the exact model name provided by the user
CAPTIONING_MODEL_NAME = "moondream/moondream-2b-2025-04-14-4bit"


def get_captioning_model(device: str = 'cpu'):
    global CAPTIONING_MODEL
    if CAPTIONING_MODEL is None:
        logging.info(
            f"[caption.py] Loading captioning model: {CAPTIONING_MODEL_NAME}..."
        )
        try:
            # Determine the device to use
            selected_device = device
            if torch.cuda.is_available():
                selected_device = "cuda"
            elif torch.backends.mps.is_available():
                selected_device = "mps"
            logging.info(
                f"[caption.py] Attempting to load captioning model on device: {selected_device}"
            )

            CAPTIONING_MODEL = AutoModelForCausalLM.from_pretrained(
                CAPTIONING_MODEL_NAME,
                trust_remote_code=True,
                device_map={"": selected_device},
            )

        except Exception as e:
            logging.error(
                f"[caption.py] Error loading captioning model {CAPTIONING_MODEL_NAME}: {e}",
                exc_info=True,
            )
            CAPTIONING_MODEL = None
    return CAPTIONING_MODEL
