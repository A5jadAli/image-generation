from datetime import datetime
from sqlalchemy import String, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid_extensions import uuid7

from app.database import Base


class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )

    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # The prompt used for generation
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # Generated image stored in S3
    image_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    image_url: Mapped[str] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationship back to user
    user: Mapped["User"] = relationship("User", back_populates="generated_images")

    def __repr__(self) -> str:
        return f"<GeneratedImage(id={self.id}, user_id={self.user_id})>"
