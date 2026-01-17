from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User
from app.schemas import UserRegisterResponse, UserResponse, FacialAttributesSchema
from app.services.face_detection import face_detection_service
from app.services.storage import storage
from app.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/users", tags=["users"])


@router.post("/register", response_model=UserRegisterResponse)
async def register_user(
    name: str = Form(..., description="User's name"),
    images: list[UploadFile] = File(..., description="4-10 photos of the user"),
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user by uploading 4-10 photos.

    The system will:
    1. Detect faces in all uploaded images
    2. Score each face for quality (size, pose, sharpness, etc.)
    3. Extract the best face crop, upper body crop, and full image
    4. Detect facial attributes (age, gender, skin tone)
    5. Create a user profile with all references

    These references will be used for all future image generations to maintain
    consistent appearance.
    """
    # Validate number of images
    if len(images) < settings.min_images_required:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least {settings.min_images_required} images are required. Got {len(images)}."
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

    # Process images to get all reference data
    user_references = face_detection_service.process_user_images(image_bytes_list)

    if user_references is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No faces detected in any of the uploaded images. Please upload clear photos with visible faces."
        )

    # Create user first to get ID
    user = User(
        name=name,
        face_image_key="",  # Will update after upload
        original_images_count=len(images),
        face_quality_score=user_references.face_quality_score,
        face_detection_confidence=user_references.detection_confidence,
        facial_attributes=user_references.attributes.to_dict(),
        face_embedding=user_references.face_embedding,
    )
    db.add(user)
    await db.flush()  # Get the user ID

    # Upload all reference images to local storage
    base_path = f"users/{user.id}"

    # 1. Face crop (required)
    face_key = f"{base_path}/face.jpg"
    await storage.upload_image(user_references.face_crop, face_key)
    user.face_image_key = face_key

    # 2. Upper body crop (optional but recommended)
    if user_references.upper_body_crop:
        upper_body_key = f"{base_path}/upper_body.jpg"
        await storage.upload_image(user_references.upper_body_crop, upper_body_key)
        user.upper_body_image_key = upper_body_key

    # 3. Full image (optional but recommended)
    if user_references.full_image:
        full_key = f"{base_path}/full.jpg"
        await storage.upload_image(user_references.full_image, full_key)
        user.full_image_key = full_key

    await db.commit()
    await db.refresh(user)

    # Build response with URLs
    face_url = await storage.get_url(user.face_image_key)
    upper_body_url = await storage.get_url(user.upper_body_image_key) if user.upper_body_image_key else None
    full_url = await storage.get_url(user.full_image_key) if user.full_image_key else None

    user_response = UserResponse(
        id=user.id,
        name=user.name,
        face_image_url=face_url,
        upper_body_image_url=upper_body_url,
        full_image_url=full_url,
        facial_attributes=FacialAttributesSchema(**user.facial_attributes) if user.facial_attributes else None,
        face_quality_score=user.face_quality_score,
        original_images_count=user.original_images_count,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )

    return UserRegisterResponse(
        user=user_response,
        face_detected=True,
        face_quality_score=user_references.face_quality_score,
        face_detection_confidence=user_references.detection_confidence,
        facial_attributes=FacialAttributesSchema(**user_references.attributes.to_dict()),
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

    # Get URLs for images
    face_url = await storage.get_url(user.face_image_key) if user.face_image_key else None
    upper_body_url = await storage.get_url(user.upper_body_image_key) if user.upper_body_image_key else None
    full_url = await storage.get_url(user.full_image_key) if user.full_image_key else None

    return UserResponse(
        id=user.id,
        name=user.name,
        face_image_url=face_url,
        upper_body_image_url=upper_body_url,
        full_image_url=full_url,
        facial_attributes=FacialAttributesSchema(**user.facial_attributes) if user.facial_attributes else None,
        face_quality_score=user.face_quality_score,
        original_images_count=user.original_images_count,
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

    # Delete all reference images from storage
    if user.face_image_key:
        await storage.delete_image(user.face_image_key)
    if user.upper_body_image_key:
        await storage.delete_image(user.upper_body_image_key)
    if user.full_image_key:
        await storage.delete_image(user.full_image_key)

    # Delete user from database (cascade will delete generated images)
    await db.delete(user)
    await db.commit()
