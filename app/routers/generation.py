from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, GeneratedImage
from app.schemas import GenerateImageRequest, GenerateImageResponse, GeneratedImageInfo
from app.services.imagen import imagen_service
from app.services.storage import storage

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("", response_model=GenerateImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate personalized images for a user.

    The system will:
    1. Fetch the user's best reference image
    2. Send it to Nano Banana with your prompt
    3. Generate an image preserving the person's facial features
    4. Store and return the generated image

    Example prompts:
    - "enjoying at a beach with sunset"
    - "working out in a modern gym"
    - "reading a book in a cozy cafe"
    - "hiking in beautiful mountains"
    - "giving a presentation in a conference room"
    - "cooking in a kitchen"
    """
    # Get user
    result = await db.execute(select(User).where(User.id == request.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {request.user_id} not found. Please register first."
        )

    # Load ALL reference images — more angles = stronger face lock
    reference_keys = user.reference_image_keys or []
    if not reference_keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No reference image found for user."
        )

    # Load primary reference (first image)
    try:
        reference_image = await storage.download_image(reference_keys[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reference image: {str(e)}"
        )

    # Load additional references for multi-angle face anchoring
    additional_references = []
    for key in reference_keys[1:]:
        try:
            img_bytes = await storage.download_image(key)
            additional_references.append(img_bytes)
        except Exception:
            continue  # Skip failed loads

    # Generate image with reference(s)
    try:
        generated_image_bytes_list = await imagen_service.generate_image(
            prompt=request.prompt,
            reference_image=reference_image,
            additional_references=additional_references if additional_references else None,
            aspect_ratio=request.aspect_ratio.value,
            number_of_images=request.number_of_images,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate image: {str(e)}"
        )

    if not generated_image_bytes_list:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No images were generated. Please try again with a different prompt."
        )

    # Store generated images and create database records
    generated_images_info = []

    for i, image_bytes in enumerate(generated_image_bytes_list):
        # Create database record
        generated_image = GeneratedImage(
            user_id=user.id,
            prompt=request.prompt,
            image_s3_key="",
        )
        db.add(generated_image)
        await db.flush()

        # Upload to storage
        storage_key = f"generated/{user.id}/{generated_image.id}.jpg"
        await storage.upload_image(image_bytes, storage_key, content_type="image/jpeg")

        # Update record
        generated_image.image_s3_key = storage_key
        image_url = await storage.get_url(storage_key)
        generated_image.image_url = image_url

        generated_images_info.append(GeneratedImageInfo(
            id=generated_image.id,
            image_url=image_url,
            prompt=request.prompt,
            created_at=generated_image.created_at,
        ))

    await db.commit()

    return GenerateImageResponse(
        user_id=user.id,
        prompt=request.prompt,
        images=generated_images_info,
    )


@router.get("/history/{user_id}", response_model=list[GeneratedImageInfo])
async def get_generation_history(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Get the generation history for a user."""
    # Check if user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found."
        )

    # Get generated images
    result = await db.execute(
        select(GeneratedImage)
        .where(GeneratedImage.user_id == user_id)
        .order_by(GeneratedImage.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    images = result.scalars().all()

    # Build response with URLs
    images_info = []
    for img in images:
        if img.image_s3_key:
            img.image_url = await storage.get_url(img.image_s3_key)
        images_info.append(GeneratedImageInfo.model_validate(img))

    return images_info
