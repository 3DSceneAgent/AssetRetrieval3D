"""
Ensure local qwen PostgreSQL database exists.
If missing, bootstrap from OSS dump.
"""
from __future__ import annotations

import tarfile
import gzip
import logging
import os
import shutil
import subprocess
from pathlib import Path

import psycopg2
import requests
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logger = logging.getLogger(__name__)


def _admin_conn_string() -> str:
    return (
        f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@"
        f"{config.DB_HOST}:{config.DB_PORT}/postgres"
    )


def _database_exists(db_name: str) -> bool:
    conn = psycopg2.connect(_admin_conn_string())
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            return cursor.fetchone() is not None
    finally:
        conn.close()


def _create_database(db_name: str) -> None:
    conn = psycopg2.connect(_admin_conn_string())
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        conn.close()


def _download_dump(url: str, target_path: Path, timeout_seconds: int) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading qwen DB dump from OSS: %s", url)

    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    logger.info("Downloaded dump to: %s", target_path)
    return target_path


def _maybe_decompress(path: Path) -> Path:
    if path.suffix != ".gz":
        return path

    decompressed = path.with_suffix("")
    logger.info("Decompressing %s -> %s", path, decompressed)
    with gzip.open(path, "rb") as src, open(decompressed, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return decompressed


def _detect_dump_format(path: Path, configured: str) -> str:
    if configured in {"custom", "plain"}:
        return configured

    # pg_dump custom format starts with "PGDMP"
    with open(path, "rb") as f:
        head = f.read(5)
    return "custom" if head == b"PGDMP" else "plain"


def _run_restore_command(db_name: str, dump_path: Path, fmt: str) -> None:
    env = os.environ.copy()
    env["PGPASSWORD"] = config.DB_PASSWORD

    base_args = [
        "-h", config.DB_HOST,
        "-p", str(config.DB_PORT),
        "-U", config.DB_USER,
        "-d", db_name,
    ]

    if fmt == "custom":
        cmd = [
            "pg_restore",
            *base_args,
            "--no-owner",
            "--no-privileges",
            str(dump_path),
        ]
    else:
        cmd = [
            "psql",
            *base_args,
            "-v", "ON_ERROR_STOP=1",
            "-f", str(dump_path),
        ]

    logger.info("Restoring qwen DB from dump (%s)", fmt)
    subprocess.run(cmd, env=env, check=True)


def _restore_from_embeddings_archive(archive_path: Path) -> None:
    extract_dir = archive_path.parent / "qwen_embeddings_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting embeddings archive: %s", archive_path)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=extract_dir)

    candidates = [extract_dir, extract_dir / "embeddings", extract_dir / "outputs" / "embeddings"]
    text_src = None
    image_src = None
    for c in candidates:
        t = c / "qwen_text_embeddings.h5"
        i = c / "qwen_image_embeddings.h5"
        if t.exists() and i.exists():
            text_src = t
            image_src = i
            break

    if text_src is None or image_src is None:
        raise RuntimeError(
            "Embeddings archive does not contain qwen_text_embeddings.h5 and qwen_image_embeddings.h5"
        )

    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_src, config.QWEN_TEXT_EMBEDDINGS_FILE)
    shutil.copy2(image_src, config.QWEN_IMAGE_EMBEDDINGS_FILE)

    logger.info("Copied embeddings to %s", config.EMBEDDINGS_DIR)
    subprocess.run([sys.executable, str(config.SCRIPTS_DIR / "04_populate_database.py")], check=True)


def _tables_exist() -> bool:
    conn = psycopg2.connect(config.DB_CONNECTION_QWEN)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('text_embeddings', 'image_embeddings')
                """
            )
            count = cursor.fetchone()[0]
            return count == 2
    finally:
        conn.close()


def ensure_qwen_db_ready() -> None:
    db_name = config.DB_NAME_QWEN

    if _database_exists(db_name):
        logger.info("Qwen database already exists: %s", db_name)
        return

    if not config.QWEN_DB_OSS_URL:
        raise RuntimeError(
            f"Database '{db_name}' not found and QWEN_DB_OSS_URL is empty. "
            "Cannot bootstrap qwen DB from OSS."
        )

    local_dump_path = Path(config.QWEN_DB_DUMP_LOCAL_PATH)
    downloaded = _download_dump(
        config.QWEN_DB_OSS_URL,
        local_dump_path,
        config.QWEN_DB_DOWNLOAD_TIMEOUT_SECONDS,
    )
    if str(downloaded).endswith((".tar.gz", ".tgz", ".tar")):
        _restore_from_embeddings_archive(downloaded)
    else:
        restored_dump = _maybe_decompress(downloaded)
        _create_database(db_name)
        dump_fmt = _detect_dump_format(restored_dump, config.QWEN_DB_DUMP_FORMAT)
        _run_restore_command(db_name, restored_dump, dump_fmt)

    if not _tables_exist():
        raise RuntimeError(
            "Qwen DB restore finished but expected tables are missing: "
            "text_embeddings/image_embeddings"
        )

    logger.info("Qwen database bootstrap completed: %s", db_name)
