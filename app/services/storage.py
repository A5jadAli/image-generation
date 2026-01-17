import aiofiles
from pathlib import Path

from app.config import get_settings

settings = get_settings()


class LocalStorage:
    """
    Local file storage for images.
    Will be replaced with S3 storage later.
    """

    def __init__(self):
        self.base_path = Path(settings.storage_path)
        self._ensure_base_path()

    def _ensure_base_path(self) -> None:
        """Ensure the base storage directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, key: str) -> Path:
        """Get the full filesystem path for a storage key."""
        return self.base_path / key

    async def upload_image(
        self,
        file_data: bytes,
        key: str,
        content_type: str = "image/jpeg"
    ) -> str:
        """Save an image to local storage and return the key."""
        full_path = self._get_full_path(key)

        # Ensure parent directories exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(full_path, "wb") as f:
            await f.write(file_data)

        return key

    async def download_image(self, key: str) -> bytes:
        """Read an image from local storage and return bytes."""
        full_path = self._get_full_path(key)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {key}")

        async with aiofiles.open(full_path, "rb") as f:
            data = await f.read()

        return data

    async def get_url(self, key: str) -> str:
        """
        Get a URL/path for accessing the file.
        For local storage, returns the relative path.
        In production with S3, this would return a presigned URL.
        """
        return f"/files/{key}"

    async def delete_image(self, key: str) -> None:
        """Delete an image from local storage."""
        full_path = self._get_full_path(key)

        if full_path.exists():
            full_path.unlink()

    async def ensure_storage_exists(self) -> None:
        """Ensure storage is ready (create directories)."""
        self._ensure_base_path()
        # Create subdirectories for faces and generated images
        (self.base_path / "faces").mkdir(exist_ok=True)
        (self.base_path / "generated").mkdir(exist_ok=True)


# Singleton instance
storage = LocalStorage()
