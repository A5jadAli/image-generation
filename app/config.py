from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # App settings
    app_name: str = "Personalized Image Generation API"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/image_gen"

    # Local storage settings (will be replaced with S3 later)
    storage_path: Path = Path("temp/uploads")

    # Nano Banana API settings
    nano_banana_api_key: str = ""

    # Image upload limits
    min_images_required: int = 1
    max_images_allowed: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
