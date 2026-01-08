"""
Global exception classes for the POPANE API
"""

from fastapi import HTTPException, status


class DatabaseNotFoundError(HTTPException):
    """Raised when the database file cannot be found"""
    def __init__(self, path: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database not found at {path}"
        )


class DatabaseConnectionError(HTTPException):
    """Raised when database connection fails"""
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {message}"
        )


class DatabaseQueryError(HTTPException):
    """Raised when a database query fails"""
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {message}"
        )
