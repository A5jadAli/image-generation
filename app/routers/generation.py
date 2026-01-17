from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, GeneratedImage
from app.schemas import GenerateImageRequest, GenerateImageResponse, GeneratedImageInfo
from app.services.imagen import imagen_service, ReferenceImages
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
    1. Fetch all stored reference images (face, upper body, full image)
    2. Get the user's facial attributes for prompt enhancement
    3. Generate images using Nano Banana with all references
    4. Store and return the generated images

    Example prompts:
    - "enjoying at a beach with sunset"
    - "working out in a modern gym"
    - "reading a book in a cozy cafe"
    - "hiking in beautiful mountains"
    - "giving a presentation in a conference room"
    """
    # Get user
    result = await db.execute(select(User).where(User.id == request.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {request.user_id} not found. Please register first."
        )

    # Load all reference images from storage
    try:
        face_image = await storage.download_image(user.face_image_key)

        upper_body_image = None
        if user.upper_body_image_key:
            upper_body_image = await storage.download_image(user.upper_body_image_key)

        full_image = None
        if user.full_image_key:
            full_image = await storage.download_image(user.full_image_key)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reference images: {str(e)}"
        )

    # Build reference images object
    reference_images = ReferenceImages(
        face_image=face_image,
        upper_body_image=upper_body_image,
        full_image=full_image,
    )

    # Get person description from attributes
    person_description = user.get_prompt_description()

    # Generate images with all references and attributes
    try:
        generated_image_bytes_list = await imagen_service.generate_image_with_references(
            prompt=request.prompt,
            reference_images=reference_images,
            person_description=person_description,
            aspect_ratio=request.aspect_ratio.value,
            number_of_images=request.number_of_images,
            negative_prompt=request.negative_prompt,
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
        # Create database record first to get ID
        generated_image = GeneratedImage(
            user_id=user.id,
            prompt=request.prompt,
            image_s3_key="",  # Will update after upload
        )
        db.add(generated_image)
        await db.flush()

        # Upload to local storage
        storage_key = f"generated/{user.id}/{generated_image.id}.jpg"
        await storage.upload_image(image_bytes, storage_key, content_type="image/jpeg")

        # Update record with storage key and URL
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

    # Refresh URLs
    images_info = []
    for img in images:
        if img.image_s3_key:
            img.image_url = await storage.get_url(img.image_s3_key)
        images_info.append(GeneratedImageInfo.model_validate(img))

    return images_info
