"""
Database utilities for PostgreSQL with pgvector.
"""
import logging
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connections and operations."""

    def __init__(self, db_name: str, connection_string: Optional[str] = None):
        self.db_name = db_name
        self.connection_string = (
            connection_string
            if connection_string
            else f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{db_name}"
        )
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None

    def create_database_if_not_exists(self):
        conn_string = (
            f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@"
            f"{config.DB_HOST}:{config.DB_PORT}/postgres"
        )

        conn = psycopg2.connect(conn_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.db_name,))
                exists = cursor.fetchone()
                if not exists:
                    logger.info("Creating database: %s", self.db_name)
                    cursor.execute(f'CREATE DATABASE "{self.db_name}"')
                else:
                    logger.info("Database already exists: %s", self.db_name)
        finally:
            conn.close()

    def enable_pgvector_extension(self):
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()

    def init_connection_pool(self, min_connections: int = 1, max_connections: int = 10):
        self.connection_pool = pool.SimpleConnectionPool(
            min_connections,
            max_connections,
            self.connection_string,
        )
        logger.info("Connection pool initialized for %s", self.db_name)

    @contextmanager
    def get_connection(self) -> Generator:
        if self.connection_pool is None:
            self.init_connection_pool()

        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    def close_pool(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Connection pool closed for %s", self.db_name)

    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = False):
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                conn.commit()

    def table_exists(self, table_name: str) -> bool:
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = %s
        )
        """
        result = self.execute_query(query, (table_name,), fetch=True)
        return result[0][0] if result else False


def get_qwen_db() -> DatabaseManager:
    return DatabaseManager(config.DB_NAME_QWEN, config.DB_CONNECTION_QWEN)


def test_connection(db_manager: DatabaseManager) -> bool:
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info("Connected to PostgreSQL: %s", version[0])
        return True
    except Exception as e:
        logger.error("Connection test failed: %s", e)
        return False
