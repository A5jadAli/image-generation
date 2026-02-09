from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, GeneratedImage
from app.models.generated_video import GeneratedVideo
from app.schemas.video import (
    GenerateVideoFromTextRequest,
    GenerateVideoFromImageRequest,
    GenerateVideoResponse,
    GeneratedVideoInfo,
)
from app.services.video import video_service
from app.services.storage import storage

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/from-text", response_model=GenerateVideoResponse)
async def generate_video_from_text(
    request: GenerateVideoFromTextRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a personalized video from a text prompt.

    The system will:
    1. Fetch the user's reference images (if use_reference_images is True)
    2. Send them to Veo 3.1 with your prompt as reference assets
    3. Generate a video preserving the person's identity
    4. Store and return the generated video

    Example prompts:
    - "walking on a beach at sunset"
    - "dancing gracefully in a modern studio"
    - "giving a presentation confidently"
    - "cooking in a beautiful kitchen"
    - "jogging through a scenic park"
    """
    # Get user
    result = await db.execute(select(User).where(User.id == request.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {request.user_id} not found. Please register first."
        )

    # Load reference images if requested
    reference_images = None
    if request.use_reference_images:
        reference_keys = user.reference_image_keys or []
        if not reference_keys:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No reference images found for user. Upload reference images first."
            )

        reference_images = []
        for key in reference_keys:
            try:
                img_bytes = await storage.download_image(key)
                reference_images.append(img_bytes)
            except Exception:
                continue  # Skip if image can't be loaded

        if not reference_images:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load reference images."
            )

    # Generate video
    try:
        video_bytes = await video_service.generate_video_from_text(
            prompt=request.prompt,
            reference_images=reference_images,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate video: {str(e)}"
        )

    # Create database record
    generated_video = GeneratedVideo(
        user_id=user.id,
        prompt=request.prompt,
        video_s3_key="",
        source_type="text",
    )
    db.add(generated_video)
    await db.flush()

    # Upload to storage
    storage_key = f"videos/{user.id}/{generated_video.id}.mp4"
    await storage.upload_image(video_bytes, storage_key, content_type="video/mp4")

    # Update record
    generated_video.video_s3_key = storage_key
    video_url = await storage.get_url(storage_key)
    generated_video.video_url = video_url

    await db.commit()

    return GenerateVideoResponse(
        user_id=user.id,
        prompt=request.prompt,
        video=GeneratedVideoInfo(
            id=generated_video.id,
            video_url=video_url,
            prompt=request.prompt,
            source_type="text",
            created_at=generated_video.created_at,
        ),
    )


@router.post("/from-image", response_model=GenerateVideoResponse)
async def generate_video_from_image(
    request: GenerateVideoFromImageRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a video from an existing generated image (image-to-video).

    The system will:
    1. Fetch the specified generated image
    2. Use it as the starting frame for video generation
    3. Animate the image according to your prompt
    4. Store and return the generated video

    Example prompts:
    - "gentle smile and slight head turn"
    - "wind blowing through hair"
    - "waving hello warmly"
    - "turning to look at camera"
    - "subtle breathing and blinking"
    """
    # Get user
    result = await db.execute(select(User).where(User.id == request.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {request.user_id} not found."
        )

    # Get the source image
    result = await db.execute(
        select(GeneratedImage).where(
            GeneratedImage.id == request.image_id,
            GeneratedImage.user_id == request.user_id,
        )
    )
    source_image = result.scalar_one_or_none()

    if source_image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generated image with ID {request.image_id} not found for this user."
        )

    # Load the source image
    try:
        source_image_bytes = await storage.download_image(source_image.image_s3_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load source image: {str(e)}"
        )

    # Generate video from image
    try:
        video_bytes = await video_service.generate_video_from_image(
            prompt=request.prompt,
            source_image=source_image_bytes,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate video: {str(e)}"
        )

    # Create database record
    generated_video = GeneratedVideo(
        user_id=user.id,
        prompt=request.prompt,
        video_s3_key="",
        source_type="image",
        source_image_id=source_image.id,
    )
    db.add(generated_video)
    await db.flush()

    # Upload to storage
    storage_key = f"videos/{user.id}/{generated_video.id}.mp4"
    await storage.upload_image(video_bytes, storage_key, content_type="video/mp4")

    # Update record
    generated_video.video_s3_key = storage_key
    video_url = await storage.get_url(storage_key)
    generated_video.video_url = video_url

    await db.commit()

    return GenerateVideoResponse(
        user_id=user.id,
        prompt=request.prompt,
        video=GeneratedVideoInfo(
            id=generated_video.id,
            video_url=video_url,
            prompt=request.prompt,
            source_type="image",
            source_image_id=source_image.id,
            created_at=generated_video.created_at,
        ),
    )


@router.get("/history/{user_id}", response_model=list[GeneratedVideoInfo])
async def get_video_history(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Get the video generation history for a user."""
    # Check if user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found."
        )

    # Get generated videos
    result = await db.execute(
        select(GeneratedVideo)
        .where(GeneratedVideo.user_id == user_id)
        .order_by(GeneratedVideo.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    videos = result.scalars().all()

    # Build response with URLs
    videos_info = []
    for vid in videos:
        if vid.video_s3_key:
            vid.video_url = await storage.get_url(vid.video_s3_key)
        videos_info.append(GeneratedVideoInfo.model_validate(vid))

    return videos_info
