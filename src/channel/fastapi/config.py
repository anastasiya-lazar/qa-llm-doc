from pydantic_settings import BaseSettings
from typing import List
import os
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document QA System"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # CORS
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

    @property
    def cors_origins_list(self) -> List[str]:
        return self.CORS_ORIGINS.split(",")

    # Rate Limiting
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    RATE_LIMIT_PER_MINUTE: int = 60

    # File Upload
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "txt", "doc", "docx"]

    # OpenAI
    OPENAI_API_KEY: str

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Storage
    UPLOAD_DIR: str = "uploads"
    STORAGE_DIR: str = "storage"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    # Get the project root directory (where .env is located)
    root_dir = Path(__file__).parent.parent.parent.parent
    env_path = root_dir / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")

    return Settings(_env_file=env_path)
