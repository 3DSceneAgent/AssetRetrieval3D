#!/usr/bin/env bash
set -euo pipefail

# Setup PostgreSQL qwen database and enable pgvector extension.

echo "Setting up PostgreSQL qwen database..."

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_NAME_QWEN="${DB_NAME_QWEN:-qwen_embeddings}"

psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d postgres -c "CREATE DATABASE \"${DB_NAME_QWEN}\";" 2>/dev/null || echo "Database ${DB_NAME_QWEN} may already exist"
psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME_QWEN" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "✓ Qwen database setup complete"
