from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User
from app.schemas import UserRegisterResponse, UserResponse
from app.services.storage import storage
from app.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/users", tags=["users"])


@router.post("/register", response_model=UserRegisterResponse)
async def register_user(
    name: str = Form(..., description="User's name"),
    images: list[UploadFile] = File(..., description="1-5 photos of the user"),
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user by uploading 1-5 photos.

    These images will be stored and used as reference when generating new images.
    The first image will be used as the primary reference for generation.
    """
    # Validate number of images
    if len(images) < settings.min_images_required:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least {settings.min_images_required} image is required."
        )

    if len(images) > settings.max_images_allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {settings.max_images_allowed} images allowed. Got {len(images)}."
        )

    # Read all images
    image_bytes_list = []
    for img in images:
        if not img.content_type or not img.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {img.filename} is not an image."
            )
        content = await img.read()
        image_bytes_list.append(content)

    # Create user
    user = User(name=name, reference_image_keys=[])
    db.add(user)
    await db.flush()

    # Store all images
    image_keys = []
    for idx, image_bytes in enumerate(image_bytes_list):
        key = f"users/{user.id}/image_{idx}.jpg"
        await storage.upload_image(image_bytes, key)
        image_keys.append(key)

    user.reference_image_keys = image_keys

    await db.commit()
    await db.refresh(user)

    # Build response with URLs
    image_urls = [await storage.get_url(key) for key in user.reference_image_keys]

    user_response = UserResponse(
        id=user.id,
        name=user.name,
        reference_image_urls=image_urls,
        images_count=len(image_keys),
        created_at=user.created_at,
        updated_at=user.updated_at,
    )

    return UserRegisterResponse(
        user=user_response,
        message=f"User registered successfully with {len(images)} image(s)",
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get user details by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found."
        )

    image_urls = []
    if user.reference_image_keys:
        image_urls = [await storage.get_url(key) for key in user.reference_image_keys]

    return UserResponse(
        id=user.id,
        name=user.name,
        reference_image_urls=image_urls,
        images_count=len(user.reference_image_keys) if user.reference_image_keys else 0,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a user and their associated images."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found."
        )

    # Delete reference images from storage
    if user.reference_image_keys:
        for key in user.reference_image_keys:
            try:
                await storage.delete_image(key)
            except Exception:
                pass

    await db.delete(user)
    await db.commit()
