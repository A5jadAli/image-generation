from pydantic import BaseModel, Field
from datetime import datetime


class UserCreate(BaseModel):
    """Schema for creating a new user (sent with the image upload)."""
    name: str = Field(..., min_length=1, max_length=255, description="User's name")


class UserResponse(BaseModel):
    """Schema for user response."""
    id: str
    name: str
    reference_image_urls: list[str] = []
    images_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserRegisterResponse(BaseModel):
    """Response after user registration."""
    user: UserResponse
    message: str = "User registered successfully"
