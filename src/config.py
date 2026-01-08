"""
Global configuration settings for the POPANE API
Uses Pydantic BaseSettings for environment variable management
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

from src.constants import Environment
import os

class Settings(BaseSettings):
    """Global application settings"""

    # Application
    APP_NAME: str = "POPANE Emotion Database API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True

    # Database
    DATABASE_PATH: str = Field(
        default=os.getenv("DATABASE_PATH", "data/popane.db"),
        description="Path to the DuckDB database file"
    )
    DATABASE_DIR: str = Field(
        default="",
        description="Optional directory containing database files"
    )

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # API
    DEFAULT_LIMIT: int = 1000
    MAX_LIMIT: int = 10000

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
