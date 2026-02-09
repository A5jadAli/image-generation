from datetime import datetime
from sqlalchemy import String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid_extension import uuid7

from app.database import Base


class GeneratedVideo(Base):
    __tablename__ = "generated_videos"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )

    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # The prompt used for generation
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # Generated video stored in storage
    video_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    video_url: Mapped[str] = mapped_column(Text, nullable=True)

    # Source type: "text" for text-to-video, "image" for image-to-video
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, default="text")

    # Optional reference to source image (for image-to-video)
    source_image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generated_images.id", ondelete="SET NULL"), nullable=True
    )

    # Video metadata
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="generated_videos")
    source_image: Mapped["GeneratedImage"] = relationship(
        "GeneratedImage", foreign_keys=[source_image_id]
    )

    def __repr__(self) -> str:
        return f"<GeneratedVideo(id={self.id}, user_id={self.user_id}, source_type={self.source_type})>"
