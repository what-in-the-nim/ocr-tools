import numpy as np
from PIL import Image
from transformers import GenerationConfig

DEFAULT_DEVICE = "cuda"
RECOGNIZER_BATCH_SIZE = 16
RECOGNIZER_USE_ACCELERATOR = False
RECOGNIZER_USE_SMART_BATCHING = True
RECOGNIZER_CONFIDENCE_THRESHOLD = 0.9
RECOGNIZER_SUPPORTED_TYPES = (str, Image.Image, np.ndarray)
RECOGNIZER_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=120,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
)

SMART_BATCHER_SUPPORTED_TYPES = (str, Image.Image, np.ndarray)
