"""Embeddings module using google/siglip-base-patch16-384 (768-dim).

Supports both image and text embeddings with the same model.
"""

import io
import logging
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
)

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, CACHE_DIR

logger = logging.getLogger(__name__)


# Lazy-loaded singleton
_model: Optional[PreTrainedModel] = None
_processor: Optional[ProcessorMixin] = None


def _load_model() -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load the SigLIP model and processor once (lazy singleton)."""
    global _model, _processor
    if _model is None or _processor is None:
        logger.info("Loading embedding model: %s ...", EMBEDDING_MODEL)
        _model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        _processor = AutoProcessor.from_pretrained(
            EMBEDDING_MODEL,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        _model.eval()
        logger.info("Model loaded successfully.")
    return _model, _processor


def get_image_embedding(image_url: str) -> Optional[list[float]]:
    """Download an image from a URL and compute its SigLIP embedding (768-dim).

    Returns a list of floats (numpy-style) or None on failure.
    """
    try:
        model, processor = _load_model()

        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embedding = outputs.squeeze().tolist()
        if isinstance(embedding, float):
            embedding = [embedding]
        # Sanity-check dimension
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(
                "Expected %d-dim embedding, got %d for %s",
                EMBEDDING_DIM,
                len(embedding),
                image_url,
            )
        return embedding
    except Exception as exc:
        logger.error("Failed to get image embedding for %s: %s", image_url, exc)
        return None


def get_text_embedding(text: str) -> Optional[list[float]]:
    """Compute a SigLIP text embedding (768-dim) from a text string.

    SigLIP uses a dual-encoder architecture; get_text_features returns
    the same 768-dim space as get_image_features.
    """
    try:
        model, processor = _load_model()

        inputs = processor(text=[text], padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        embedding = outputs.squeeze().tolist()
        if isinstance(embedding, float):
            embedding = [embedding]
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(
                "Expected %d-dim text embedding, got %d",
                EMBEDDING_DIM,
                len(embedding),
            )
        return embedding
    except Exception as exc:
        logger.error("Failed to get text embedding: %s", exc)
        return None


def unload_model() -> None:
    """Free GPU memory by deleting the model."""
    global _model, _processor
    _model = None
    _processor = None
    torch.cuda.empty_cache()
