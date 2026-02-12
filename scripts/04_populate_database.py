"""
Create qwen database schema and populate embeddings.
"""
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from psycopg2.extras import execute_values

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.data_loader import DataLoader
from utils.db_utils import DatabaseManager, get_qwen_db
from utils.h5_utils import load_multimodal_text_embeddings_h5, load_text_embeddings_h5

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DatabasePopulator:
    def __init__(self):
        self.data_loader = DataLoader()
        self.high_quality_ids = self._load_quality_flags()
        self.valid_asset_ids = self._load_valid_assets()

    def _load_valid_assets(self) -> set:
        if not config.INDEX_MAPPING_FILE.exists():
            logger.warning("Index mapping file not found: %s", config.INDEX_MAPPING_FILE)
            return set()
        with open(config.INDEX_MAPPING_FILE, "r") as f:
            data = json.load(f)
        logger.info("Found %s valid assets", len(data))
        return set(data.keys())

    def _load_quality_flags(self) -> set:
        if not config.HIGH_QUALITY_ASSETS_FILE.exists():
            logger.warning("High quality assets file not found: %s", config.HIGH_QUALITY_ASSETS_FILE)
            return set()
        with open(config.HIGH_QUALITY_ASSETS_FILE, "r") as f:
            ids = set(json.load(f))
        logger.info("Found %s high quality assets", len(ids))
        return ids

    def create_qwen_tables(self, db: DatabaseManager, embedding_dim: int):
        logger.info("Creating Qwen tables (embedding_dim=%s)", embedding_dim)

        text_table_sql = f"""
        CREATE TABLE IF NOT EXISTS text_embeddings (
            asset_id VARCHAR(255) PRIMARY KEY,
            english_embedding vector({embedding_dim}),
            chinese_embedding vector({embedding_dim}),
            is_high_quality BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        image_table_sql = f"""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            asset_id VARCHAR(255) PRIMARY KEY,
            embedding vector({embedding_dim}),
            num_images INTEGER,
            is_high_quality BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        db.execute_query(text_table_sql)
        db.execute_query(image_table_sql)

        db.execute_query("ALTER TABLE text_embeddings ADD COLUMN IF NOT EXISTS is_high_quality BOOLEAN DEFAULT FALSE;")
        db.execute_query("ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS is_high_quality BOOLEAN DEFAULT FALSE;")
        db.execute_query("CREATE INDEX IF NOT EXISTS text_is_high_quality_idx ON text_embeddings (is_high_quality);")
        db.execute_query("CREATE INDEX IF NOT EXISTS image_is_high_quality_idx ON image_embeddings (is_high_quality);")

    def create_vector_indexes(self, db: DatabaseManager):
        db.execute_query(
            """
            CREATE INDEX IF NOT EXISTS text_english_embedding_idx
            ON text_embeddings
            USING ivfflat (english_embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
        db.execute_query(
            """
            CREATE INDEX IF NOT EXISTS text_chinese_embedding_idx
            ON text_embeddings
            USING ivfflat (chinese_embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
        db.execute_query(
            """
            CREATE INDEX IF NOT EXISTS image_embedding_idx
            ON image_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )

    def insert_qwen_text_embeddings(self, db: DatabaseManager, filepath: Path):
        logger.info("Loading Qwen text embeddings from %s", filepath)
        en_dict, cn_dict = load_multimodal_text_embeddings_h5(filepath)
        all_asset_ids = set(en_dict.keys()) | set(cn_dict.keys())

        data_to_insert = []
        for asset_id in all_asset_ids:
            if asset_id not in self.valid_asset_ids:
                continue
            is_hq = asset_id in self.high_quality_ids

            en_emb = en_dict.get(asset_id)
            cn_emb = cn_dict.get(asset_id)

            en_list = en_emb.tolist() if en_emb is not None and not np.all(en_emb == 0) else None
            cn_list = cn_emb.tolist() if cn_emb is not None and not np.all(cn_emb == 0) else None

            data_to_insert.append((asset_id, en_list, cn_list, is_hq))

        with db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(
                    cursor,
                    """
                    INSERT INTO text_embeddings (asset_id, english_embedding, chinese_embedding, is_high_quality)
                    VALUES %s
                    ON CONFLICT (asset_id) DO UPDATE
                    SET english_embedding = COALESCE(EXCLUDED.english_embedding, text_embeddings.english_embedding),
                        chinese_embedding = COALESCE(EXCLUDED.chinese_embedding, text_embeddings.chinese_embedding),
                        is_high_quality = EXCLUDED.is_high_quality
                    """,
                    data_to_insert,
                    page_size=2000,
                )
                conn.commit()
                logger.info("Inserted %s Qwen text embeddings", len(data_to_insert))
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def insert_qwen_image_embeddings(self, db: DatabaseManager, filepath: Path):
        logger.info("Loading Qwen image embeddings from %s", filepath)
        embeddings_dict = load_text_embeddings_h5(filepath)

        data_to_insert = []
        for asset_id, embedding in embeddings_dict.items():
            if asset_id not in self.valid_asset_ids:
                continue
            is_hq = asset_id in self.high_quality_ids
            data_to_insert.append((asset_id, embedding.tolist(), config.QWEN_NUM_IMAGES, is_hq))

        with db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(
                    cursor,
                    """
                    INSERT INTO image_embeddings (asset_id, embedding, num_images, is_high_quality)
                    VALUES %s
                    ON CONFLICT (asset_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        num_images = EXCLUDED.num_images,
                        is_high_quality = EXCLUDED.is_high_quality
                    """,
                    data_to_insert,
                    page_size=2000,
                )
                conn.commit()
                logger.info("Inserted %s Qwen image embeddings", len(data_to_insert))
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def populate_qwen_database(self):
        logger.info("=== Populating Qwen Database ===")

        with h5py.File(config.QWEN_TEXT_EMBEDDINGS_FILE, "r") as f:
            embedding_dim = f.attrs["embedding_dim"]

        db = get_qwen_db()
        db.create_database_if_not_exists()
        db.enable_pgvector_extension()

        self.create_qwen_tables(db, int(embedding_dim))
        self.insert_qwen_text_embeddings(db, config.QWEN_TEXT_EMBEDDINGS_FILE)
        self.insert_qwen_image_embeddings(db, config.QWEN_IMAGE_EMBEDDINGS_FILE)
        self.create_vector_indexes(db)

        db.close_pool()
        logger.info("✓ Qwen database populated successfully")


def main():
    try:
        if not config.QWEN_TEXT_EMBEDDINGS_FILE.exists() or not config.QWEN_IMAGE_EMBEDDINGS_FILE.exists():
            logger.error("Qwen embeddings not found. Please run scripts/03_embed_qwen.py first")
            sys.exit(1)

        populator = DatabasePopulator()
        populator.populate_qwen_database()
    except Exception as e:
        logger.error("Database population failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
