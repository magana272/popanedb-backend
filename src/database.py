"""
Database connection management for DuckDB
"""

import os
from contextlib import contextmanager
from typing import Generator

import duckdb

from src.config import settings
from src.exceptions import DatabaseNotFoundError, DatabaseConnectionError


def validate_database_exists() -> bool:
    """Check if the database file exists"""
    return os.path.exists(settings.DATABASE_PATH)


@contextmanager
def get_db_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager for database connections.
    Ensures connections are properly closed after use.

    Usage:
        with get_db_connection() as conn:
            result = conn.execute("SELECT * FROM table").fetchall()
    """
    if not validate_database_exists():
        raise DatabaseNotFoundError(settings.DATABASE_PATH)

    conn = None
    try:
        conn = duckdb.connect(database=settings.DATABASE_PATH, read_only=True)
        yield conn
    except Exception as e:
        raise DatabaseConnectionError(str(e))
    finally:
        if conn:
            conn.close()


def get_db() -> duckdb.DuckDBPyConnection:
    """
    Get a database connection (non-context manager version).
    Caller is responsible for closing the connection.

    Note: Prefer using get_db_connection() context manager when possible.
    """
    if not validate_database_exists():
        raise DatabaseNotFoundError(settings.DATABASE_PATH)

    try:
        return duckdb.connect(database=settings.DATABASE_PATH, read_only=True)
    except Exception as e:
        raise DatabaseConnectionError(str(e))
