#!/usr/bin/env bash
set -euo pipefail

# Export qwen_embeddings database for OSS upload.

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_NAME_QWEN="${DB_NAME_QWEN:-qwen_embeddings}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/db_exports}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DUMP_PATH="${DUMP_PATH:-${OUTPUT_DIR}/${DB_NAME_QWEN}_${TIMESTAMP}.dump}"
GZIP_OUTPUT="${GZIP_OUTPUT:-true}"

mkdir -p "$OUTPUT_DIR"

echo "Exporting database ${DB_NAME_QWEN} from ${DB_HOST}:${DB_PORT} ..."
pg_dump \
  -h "$DB_HOST" \
  -p "$DB_PORT" \
  -U "$DB_USER" \
  -d "$DB_NAME_QWEN" \
  -Fc \
  --no-owner \
  --no-privileges \
  -f "$DUMP_PATH"

if [[ "$GZIP_OUTPUT" == "true" ]]; then
  gzip -f "$DUMP_PATH"
  DUMP_PATH="${DUMP_PATH}.gz"
fi

echo "Done. Export file: $DUMP_PATH"
echo "You can upload this file to Alibaba Cloud OSS and set QWEN_DB_OSS_URL." 
