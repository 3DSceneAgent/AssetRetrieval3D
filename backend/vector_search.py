"""
Vector search functionality for Qwen pgvector database.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db_utils import get_qwen_db
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class VectorSearch:
    """Handles vector similarity search in Qwen pgvector database."""

    def __init__(self):
        self.qwen_db = get_qwen_db()
        self.data_loader = DataLoader()
        self.captions_en = self.data_loader.load_english_captions()
        self.captions_cn = self.data_loader.load_chinese_captions()

    def search_text_embeddings(
        self,
        query_embedding: np.ndarray,
        language: str = "english",
        top_k: int = 10,
        high_quality_only: bool = False,
    ) -> List[Tuple[str, float]]:
        embedding_column = "english_embedding" if language == "english" else "chinese_embedding"
        embedding_list = query_embedding.tolist()

        where_clause = f"{embedding_column} IS NOT NULL"
        if high_quality_only:
            where_clause += " AND is_high_quality = TRUE"

        query = f"""
        SELECT asset_id, 1 - ({embedding_column} <=> %s::vector) AS similarity
        FROM text_embeddings
        WHERE {where_clause}
        ORDER BY {embedding_column} <=> %s::vector
        LIMIT %s
        """

        with self.qwen_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (embedding_list, embedding_list, top_k))
            results = cursor.fetchall()
            cursor.close()

        return [(asset_id, float(similarity)) for asset_id, similarity in results]

    def search_image_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        high_quality_only: bool = False,
    ) -> List[Tuple[str, float]]:
        embedding_list = query_embedding.tolist()
        where_clause = "WHERE is_high_quality = TRUE" if high_quality_only else ""

        query = f"""
        SELECT asset_id, 1 - (embedding <=> %s::vector) AS similarity
        FROM image_embeddings
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        with self.qwen_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (embedding_list, embedding_list, top_k))
            results = cursor.fetchall()
            cursor.close()

        return [(asset_id, float(similarity)) for asset_id, similarity in results]

    def search(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        language: str = "english",
        cross_modal: bool = False,
        top_k: int = 10,
        high_quality_only: bool = False,
    ) -> List[Dict]:
        if query_type == "text":
            results = (
                self.search_image_embeddings(query_embedding, top_k, high_quality_only)
                if cross_modal
                else self.search_text_embeddings(query_embedding, language, top_k, high_quality_only)
            )
        elif query_type == "image":
            results = (
                self.search_text_embeddings(query_embedding, language, top_k, high_quality_only)
                if cross_modal
                else self.search_image_embeddings(query_embedding, top_k, high_quality_only)
            )
        else:
            raise ValueError(f"Unsupported query_type: {query_type}")

        enriched_results = []
        for asset_id, similarity in results:
            enriched_results.append(
                {
                    "asset_id": asset_id,
                    "similarity": similarity,
                    "caption_en": self.captions_en.get(asset_id, ""),
                    "caption_cn": self.captions_cn.get(asset_id, ""),
                    "objaverse_id": self.data_loader.get_objaverse_id(asset_id),
                }
            )

        return enriched_results

    def close(self):
        self.qwen_db.close_pool()
