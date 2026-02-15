"""
Central configuration for the 3D Asset Retrieval System.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ==================== Project Paths ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data files
CAPTIONS_FILE = DATA_DIR / "text_captions_cap3d.json"
CAPTIONS_CN_FILE = DATA_DIR / "text_captions_cap3d_cn.json"
GOBJAVERSE_DIR = DATA_DIR / "gobjaverse"
INDEX_MAPPING_FILE = DATA_DIR / "gobjaverse_index_to_objaverse.json"
INDEX_MAPPING_FILE_WITH_IMAGE = DATA_DIR / "gobjaverse_280k_index_to_objaverse.json"
HIGH_QUALITY_ASSETS_FILE = DATA_DIR / "kiui_gobj_merged.json"

# Output directories
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
TRANSLATION_DIR = OUTPUTS_DIR / "translations"
BATCH_JSONL_DIR = OUTPUTS_DIR / "batch_jsonl"

# Create output directories
for dir_path in [OUTPUTS_DIR, EMBEDDINGS_DIR, TRANSLATION_DIR, BATCH_JSONL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Qwen embeddings output (HDF5 format)
QWEN_TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "qwen_text_embeddings.h5"
QWEN_IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "qwen_image_embeddings.h5"

# ==================== API Configuration ====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TRANSLATION_MODEL = "qwen-plus"
QWEN_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"

# ==================== Database Configuration ====================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DB_NAME_QWEN = os.getenv("DB_NAME_QWEN", "qwen_embeddings")
DB_CONNECTION_QWEN = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_QWEN}"
)

# Auto-bootstrap Qwen DB from OSS dump when DB is missing.
QWEN_DB_AUTO_BOOTSTRAP = os.getenv("QWEN_DB_AUTO_BOOTSTRAP", "true").lower() == "true"
QWEN_DB_OSS_URL = os.getenv("QWEN_DB_OSS_URL", "https://fishwowater.oss-cn-shenzhen.aliyuncs.com/qwen_embeddings_20260212_170200.dump.gz")
QWEN_DB_DUMP_FORMAT = os.getenv("QWEN_DB_DUMP_FORMAT", "auto")  # auto|custom|plain
QWEN_DB_DUMP_LOCAL_PATH = os.getenv("QWEN_DB_DUMP_LOCAL_PATH", "/tmp/qwen_embeddings.dump")
QWEN_DB_REUSE_LOCAL_DUMP = os.getenv("QWEN_DB_REUSE_LOCAL_DUMP", "true").lower() == "true"
QWEN_DB_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("QWEN_DB_DOWNLOAD_TIMEOUT_SECONDS", "1800"))

# ==================== Processing Configuration ====================
MAX_ASSETS = int(os.getenv("MAX_ASSETS", "0")) or None
TRANSLATION_BATCH_SIZE = 1000
EMBEDDING_BATCH_SIZE = 256
TEXT_EMBEDDING_API_BATCH_SIZE = 1000
QWEN_NUM_IMAGES = 8

# ==================== Backend/Frontend Configuration ====================
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8002"))
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "7864"))

BASE_URL_TEMPLATE = os.getenv(
    "BASE_URL_TEMPLATE",
    "https://placeholder.com/models/{objaverse_id}.glb",
)

# ==================== Search Configuration ====================
SIMILARITY_METRIC = "cosine"
DEFAULT_TOP_K = 10
MAX_TOP_K = 100

# ==================== Logging Configuration ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ==================== Helper Functions ====================
def get_asset_image_dir(gobjaverse_id: str) -> Path:
    """Get the directory path for a given gobjaverse ID."""
    return GOBJAVERSE_DIR / gobjaverse_id


def validate_config():
    """Validate that all required configuration is present."""
    errors = []

    if not DASHSCOPE_API_KEY:
        errors.append("DASHSCOPE_API_KEY environment variable not set")

    if not CAPTIONS_FILE.exists():
        errors.append(f"Captions file not found: {CAPTIONS_FILE}")

    if not GOBJAVERSE_DIR.exists():
        errors.append(f"Gobjaverse directory not found: {GOBJAVERSE_DIR}")

    if not INDEX_MAPPING_FILE.exists():
        errors.append(f"Index mapping file not found: {INDEX_MAPPING_FILE}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


if __name__ == "__main__":
    try:
        validate_config()
        print("✓ Configuration is valid")
        print(f"  - Project root: {PROJECT_ROOT}")
        print(f"  - Max assets: {MAX_ASSETS or 'All'}")
        print(f"  - Qwen model: {QWEN_EMBEDDING_MODEL}")
    except ValueError as e:
        print(f"✗ {e}")
