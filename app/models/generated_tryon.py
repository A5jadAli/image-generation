from datetime import datetime
from sqlalchemy import String, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid_extension import uuid7

from app.database import Base


class GeneratedTryon(Base):
    __tablename__ = "generated_tryons"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )

    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # The clothing description / prompt
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Clothing image stored in storage
    clothing_image_key: Mapped[str] = mapped_column(String(512), nullable=False)
    clothing_image_url: Mapped[str] = mapped_column(Text, nullable=True)

    # Generated try-on result image
    result_image_key: Mapped[str] = mapped_column(String(512), nullable=False)
    result_image_url: Mapped[str] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationship back to user
    user: Mapped["User"] = relationship("User", back_populates="generated_tryons")

    def __repr__(self) -> str:
        return f"<GeneratedTryon(id={self.id}, user_id={self.user_id})>"
