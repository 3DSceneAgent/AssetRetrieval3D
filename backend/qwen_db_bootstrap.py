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
from urllib.parse import urlparse

import psycopg2
import requests
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logger = logging.getLogger(__name__)
_TX_TIMEOUT_ERROR = 'unrecognized configuration parameter "transaction_timeout"'


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


def _drop_database(db_name: str) -> None:
    conn = psycopg2.connect(_admin_conn_string())
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s
                  AND pid <> pg_backend_pid()
                """,
                (db_name,),
            )
            cursor.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        conn.close()


def _download_dump(
    url: str,
    target_path: Path,
    timeout_seconds: int,
    reuse_local: bool = True,
) -> Path:
    if target_path.exists() and target_path.is_dir():
        raise RuntimeError(
            f"QWEN_DB_DUMP_LOCAL_PATH points to a directory, expected a file path: {target_path}"
        )
    if reuse_local and target_path.exists():
        size = target_path.stat().st_size
        if size > 0:
            logger.info("Using cached qwen DB dump: %s (%d bytes)", target_path, size)
            return target_path
        logger.warning("Cached qwen DB dump is empty. Re-downloading: %s", target_path)
        target_path.unlink(missing_ok=True)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.parent / f"{target_path.name}.part"
    temp_path.unlink(missing_ok=True)
    logger.info("Downloading qwen DB dump from remote URL: %s", url)

    try:
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            expected_size = int(response.headers.get("Content-Length", "0")) or None
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if not temp_path.exists():
        raise RuntimeError(f"Download finished but target file is missing: {temp_path}")

    size = temp_path.stat().st_size
    if size <= 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {temp_path}")
    if expected_size is not None and size != expected_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded file size mismatch: "
            f"expected {expected_size} bytes, got {size} bytes ({temp_path})"
        )

    temp_path.replace(target_path)
    logger.info("Downloaded dump to: %s", target_path)
    return target_path


def _is_tar_archive(path: Path) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except OSError:
        return False


def _is_gzip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _looks_like_tar_url(url: str) -> bool:
    parsed = urlparse(url)
    lower = parsed.path.lower()
    return lower.endswith((".tar.gz", ".tgz", ".tar"))


def _maybe_decompress(path: Path) -> Path:
    if not _is_gzip_file(path):
        return path

    decompressed = path.with_suffix("")
    logger.info("Decompressing %s -> %s", path, decompressed)
    try:
        with gzip.open(path, "rb") as src, open(decompressed, "wb") as dst:
            shutil.copyfileobj(src, dst)
    except (OSError, EOFError) as e:
        if decompressed.exists():
            decompressed.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to decompress gzip file {path}: {e}. "
            "The downloaded artifact may be truncated or corrupted."
        ) from e
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
    if fmt != "custom":
        subprocess.run(cmd, env=env, check=True)
        return

    completed = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if completed.returncode == 0:
        return

    combined_error = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    if _TX_TIMEOUT_ERROR in combined_error:
        logger.warning(
            "Direct pg_restore failed with transaction_timeout incompatibility. "
            "Retrying via SQL stream filter for compatibility."
        )
        _run_custom_restore_with_compat_filter(base_args, dump_path, env)
        return

    raise subprocess.CalledProcessError(
        completed.returncode,
        cmd,
        output=completed.stdout,
        stderr=completed.stderr,
    )


def _run_custom_restore_with_compat_filter(
    base_args: list[str],
    dump_path: Path,
    env: dict[str, str],
) -> None:
    restore_cmd = [
        "pg_restore",
        "--no-owner",
        "--no-privileges",
        str(dump_path),
    ]
    psql_cmd = [
        "psql",
        *base_args,
        "-v", "ON_ERROR_STOP=1",
    ]

    skipped = 0
    restore_proc = subprocess.Popen(
        restore_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    psql_proc = subprocess.Popen(
        psql_cmd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert restore_proc.stdout is not None
        assert psql_proc.stdin is not None

        for line in restore_proc.stdout:
            if line.strip().startswith("SET transaction_timeout"):
                skipped += 1
                continue
            psql_proc.stdin.write(line)

        psql_proc.stdin.close()
        restore_stderr = restore_proc.stderr.read() if restore_proc.stderr else ""
        restore_rc = restore_proc.wait()
        psql_stderr = psql_proc.stderr.read() if psql_proc.stderr else ""
        psql_rc = psql_proc.wait()
    finally:
        if restore_proc.stdout:
            restore_proc.stdout.close()
        if restore_proc.stderr:
            restore_proc.stderr.close()
        if psql_proc.stdin and not psql_proc.stdin.closed:
            psql_proc.stdin.close()
        if psql_proc.stderr:
            psql_proc.stderr.close()

    if restore_rc != 0:
        raise RuntimeError(
            "pg_restore failed during compatibility fallback.\n"
            f"Command: {' '.join(restore_cmd)}\n"
            f"stderr: {restore_stderr.strip()}"
        )

    if psql_rc != 0:
        raise RuntimeError(
            "psql failed during compatibility fallback.\n"
            f"Command: {' '.join(psql_cmd)}\n"
            f"stderr: {psql_stderr.strip()}"
        )

    logger.info(
        "Compatibility restore completed. Skipped %d incompatible transaction_timeout statements.",
        skipped,
    )


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
        if _tables_exist():
            logger.info("Qwen database already exists and required tables are present: %s", db_name)
            return
        logger.warning(
            "Qwen database exists but required tables are missing. "
            "Will recreate and restore from OSS: %s",
            db_name,
        )
        _drop_database(db_name)

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
        reuse_local=config.QWEN_DB_REUSE_LOCAL_DUMP,
    )

    if _is_tar_archive(downloaded) or _looks_like_tar_url(config.QWEN_DB_OSS_URL):
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
