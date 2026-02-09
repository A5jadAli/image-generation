from pydantic import BaseModel, Field
from datetime import datetime


class GenerateVideoFromTextRequest(BaseModel):
    """Request to generate a video from text prompt."""
    user_id: str = Field(..., description="The user ID to generate video for")
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Description of the video scene/action (e.g., 'walking on a beach at sunset', 'dancing in a studio')"
    )
    use_reference_images: bool = Field(
        default=True,
        description="Whether to use the user's reference images for personalization"
    )


class GenerateVideoFromImageRequest(BaseModel):
    """Request to generate a video from an existing generated image."""
    user_id: str = Field(..., description="The user ID")
    image_id: str = Field(..., description="The ID of a generated image to animate")
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Description of the motion/action for the video (e.g., 'gentle smile and head turn', 'waving hello')"
    )


class GeneratedVideoInfo(BaseModel):
    """Info about a single generated video."""
    id: str
    video_url: str
    prompt: str
    source_type: str
    source_image_id: str | None = None
    duration_seconds: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class GenerateVideoResponse(BaseModel):
    """Response with generated video."""
    user_id: str
    prompt: str
    video: GeneratedVideoInfo
    message: str = "Video generated successfully"
