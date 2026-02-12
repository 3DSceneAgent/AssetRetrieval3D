#!/usr/bin/env bash
set -euo pipefail

# Package qwen embeddings for OSS upload.

EMBED_DIR="${EMBED_DIR:-./outputs/embeddings}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/embedding_exports}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUTPUT_DIR}/qwen_embeddings_${TIMESTAMP}.tar.gz}"

TEXT_FILE="${EMBED_DIR}/qwen_text_embeddings.h5"
IMAGE_FILE="${EMBED_DIR}/qwen_image_embeddings.h5"

if [[ ! -f "$TEXT_FILE" || ! -f "$IMAGE_FILE" ]]; then
  echo "Missing embeddings files:" >&2
  echo "  $TEXT_FILE" >&2
  echo "  $IMAGE_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

tar -C "$EMBED_DIR" -czf "$ARCHIVE_PATH" qwen_text_embeddings.h5 qwen_image_embeddings.h5

echo "Done. Embeddings archive: $ARCHIVE_PATH"
echo "Upload this archive to OSS and set QWEN_DB_OSS_URL to the file URL."
