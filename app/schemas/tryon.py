from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TryonAspectRatio(str, Enum):
    PORTRAIT_3_4 = "3:4"
    PORTRAIT_9_16 = "9:16"
    SQUARE = "1:1"
    LANDSCAPE_4_3 = "4:3"


class VirtualTryonRequest(BaseModel):
    """Request for virtual try-on generation."""
    user_id: str = Field(..., description="The user ID whose face/body to use")
    clothing_description: str = Field(
        default="",
        max_length=1000,
        description=(
            "Optional description of the clothing or desired look. "
            "Examples: 'casual summer outfit', 'formal business suit', "
            "'red evening dress'. Leave empty to let the AI infer from the image."
        ),
    )
    aspect_ratio: TryonAspectRatio = Field(
        default=TryonAspectRatio.PORTRAIT_3_4,
        description="Aspect ratio of the output (3:4 recommended for fashion)",
    )
    number_of_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of try-on variations to generate",
    )


class GeneratedTryonInfo(BaseModel):
    """Info about a single try-on result."""
    id: str
    clothing_image_url: str
    result_image_url: str
    prompt: str
    created_at: datetime

    class Config:
        from_attributes = True


class VirtualTryonResponse(BaseModel):
    """Response with try-on results."""
    user_id: str
    clothing_description: str
    results: list[GeneratedTryonInfo]
    message: str = "Virtual try-on generated successfully"
