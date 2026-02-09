from datetime import datetime
from sqlalchemy import String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid_extension import uuid7

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Reference images stored as JSON array of storage keys
    # e.g., ["users/abc123/image_0.jpg", "users/abc123/image_1.jpg"]
    reference_image_keys: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationship to generated images
    generated_images: Mapped[list["GeneratedImage"]] = relationship(
        "GeneratedImage", back_populates="user", cascade="all, delete-orphan"
    )

    # Relationship to generated videos
    generated_videos: Mapped[list["GeneratedVideo"]] = relationship(
        "GeneratedVideo", back_populates="user", cascade="all, delete-orphan"
    )

    # Relationship to generated try-ons
    generated_tryons: Mapped[list["GeneratedTryon"]] = relationship(
        "GeneratedTryon", back_populates="user", cascade="all, delete-orphan"
    )

    def get_reference_key(self) -> str | None:
        """Get the first reference image key for generation."""
        if not self.reference_image_keys:
            return None
        return self.reference_image_keys[0]

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name={self.name})>"
