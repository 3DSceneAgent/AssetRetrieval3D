"""
Service for generating embeddings from queries (Qwen only).
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

import dashscope
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating query embeddings via Qwen API."""

    def embed_text_qwen(self, text: str) -> np.ndarray:
        input_data = [{"text": text}]
        resp = dashscope.MultiModalEmbedding.call(
            api_key=config.DASHSCOPE_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL,
            input=input_data,
        )

        if resp.status_code == 200 and resp.output:
            embeddings = resp.output.get("embeddings", [])
            if embeddings:
                return np.array(embeddings[0]["embedding"], dtype=np.float32)

        raise RuntimeError("Failed to embed text with Qwen API")

    def embed_image_qwen(self, image_path: str) -> np.ndarray:
        input_data = [{"image": image_path}]
        resp = dashscope.MultiModalEmbedding.call(
            api_key=config.DASHSCOPE_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL,
            input=input_data,
        )

        if resp.status_code == 200 and resp.output:
            embeddings = resp.output.get("embeddings", [])
            if embeddings:
                return np.array(embeddings[0]["embedding"], dtype=np.float32)

        raise RuntimeError("Failed to embed image with Qwen API")

    def embed_query(
        self,
        query: Optional[str] = None,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
    ) -> np.ndarray:
        if query is not None:
            return self.embed_text_qwen(query)

        if image_path is not None:
            return self.embed_image_qwen(image_path)

        if image is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                image.save(tmp.name)
                return self.embed_image_qwen(tmp.name)

        raise ValueError("Either query, image_path, or image must be provided")
