from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AspectRatio(str, Enum):
    SQUARE = "1:1"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_3_4 = "3:4"


class GenerateImageRequest(BaseModel):
    """Request to generate an image for a user."""
    user_id: str = Field(..., description="The user ID to generate image for")
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Description of the scene/activity (e.g., 'enjoying at a beach', 'working out in gym')"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="Aspect ratio of the generated image"
    )
    number_of_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate"
    )


class GeneratedImageInfo(BaseModel):
    """Info about a single generated image."""
    id: str
    image_url: str
    prompt: str
    created_at: datetime

    class Config:
        from_attributes = True


class GenerateImageResponse(BaseModel):
    """Response with generated images."""
    user_id: str
    prompt: str
    images: list[GeneratedImageInfo]
    message: str = "Images generated successfully"
