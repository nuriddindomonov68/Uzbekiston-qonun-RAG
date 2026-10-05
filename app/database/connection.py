import sqlite3
import os
from contextlib import contextmanager
from typing import Generator
from app.utils.logging import get_logger
from app.core.config import settings

logger = get_logger("database.connection")


def _db_path() -> str:
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    return settings.DB_PATH


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"Database error: {exc}")
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they do not already exist."""
    logger.info(f"Initialising SQLite database at {_db_path()} …")
    with get_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings   TEXT
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                filename      TEXT NOT NULL,
                filepath      TEXT NOT NULL,
                file_type     TEXT NOT NULL,
                row_count     INTEGER NOT NULL,
                column_count  INTEGER NOT NULL,
                schema_json   TEXT NOT NULL,
                uploaded_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS models (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                dataset_id    TEXT NOT NULL,
                model_type    TEXT NOT NULL,
                algorithm     TEXT NOT NULL,
                target_column TEXT,
                features      TEXT NOT NULL,
                metrics       TEXT NOT NULL,
                filepath      TEXT NOT NULL,
                trained_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS charts (
                id         TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                dataset_id TEXT,
                title      TEXT,
                chart_type TEXT NOT NULL,
                filepath   TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)
    logger.info("Database initialised successfully.")
