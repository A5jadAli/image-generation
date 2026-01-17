from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid7 import uuid7

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Multiple reference images stored locally (keys/paths)
    face_image_key: Mapped[str] = mapped_column(String(512), nullable=False)
    upper_body_image_key: Mapped[str] = mapped_column(String(512), nullable=True)
    full_image_key: Mapped[str] = mapped_column(String(512), nullable=True)

    # Facial attributes as JSON
    # {estimated_age, gender, has_glasses, has_beard, skin_tone}
    facial_attributes: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Face embedding (512-dimensional vector) for identity matching
    face_embedding: Mapped[list] = mapped_column(JSON, nullable=True)

    # Quality metrics
    face_quality_score: Mapped[float] = mapped_column(Float, nullable=True)
    face_detection_confidence: Mapped[float] = mapped_column(Float, nullable=True)

    # Original images count
    original_images_count: Mapped[int] = mapped_column(Integer, default=0)

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

    def get_prompt_description(self) -> str:
        """Generate a description from attributes for prompt enhancement."""
        if not self.facial_attributes:
            return ""

        parts = []
        attrs = self.facial_attributes

        if attrs.get("gender"):
            parts.append(attrs["gender"])

        age = attrs.get("estimated_age")
        if age:
            if age < 25:
                parts.append("young adult")
            elif age < 40:
                parts.append("adult")
            elif age < 60:
                parts.append("middle-aged")
            else:
                parts.append("senior")

        if attrs.get("has_beard"):
            parts.append("with beard")

        if attrs.get("has_glasses"):
            parts.append("wearing glasses")

        return " ".join(parts) if parts else ""

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name={self.name})>"
