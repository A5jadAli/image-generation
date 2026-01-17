from pydantic import BaseModel, Field
from datetime import datetime


class FacialAttributesSchema(BaseModel):
    """Schema for facial attributes."""
    estimated_age: int | None = None
    gender: str | None = None
    has_glasses: bool = False
    has_beard: bool = False
    skin_tone: str | None = None


class UserCreate(BaseModel):
    """Schema for creating a new user (sent with the image upload)."""
    name: str = Field(..., min_length=1, max_length=255, description="User's name")


class UserResponse(BaseModel):
    """Schema for user response."""
    id: str
    name: str
    face_image_url: str | None = None
    upper_body_image_url: str | None = None
    full_image_url: str | None = None
    facial_attributes: FacialAttributesSchema | None = None
    face_quality_score: float | None = None
    original_images_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserRegisterResponse(BaseModel):
    """Response after user registration with face extraction."""
    user: UserResponse
    message: str = "User registered successfully with face and body references extracted"
    face_detected: bool = True
    face_quality_score: float
    face_detection_confidence: float
    facial_attributes: FacialAttributesSchema | None = None
