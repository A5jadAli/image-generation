from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.database import get_db
from app.models import User, GeneratedTryon
from app.schemas.tryon import (
    VirtualTryonResponse,
    GeneratedTryonInfo,
    TryonAspectRatio,
)
from app.services.imagen import imagen_service
from app.services.storage import storage

router = APIRouter(prefix="/tryon", tags=["virtual-tryon"])


@router.post("", response_model=VirtualTryonResponse)
async def virtual_tryon(
    user_id: str = Form(..., description="The user ID whose face/body to use"),
    clothing_image: UploadFile = File(
        ..., description="Photo of the clothing item to try on"
    ),
    clothing_description: str = Form(
        default="",
        description=(
            "Optional description of the clothing or styling notes. "
            "Examples: 'casual summer dress', 'formal navy suit with tie'"
        ),
    ),
    aspect_ratio: TryonAspectRatio = Form(
        default=TryonAspectRatio.PORTRAIT_3_4,
        description="Output aspect ratio (3:4 recommended for fashion)",
    ),
    number_of_images: int = Form(
        default=1,
        ge=1,
        le=4,
        description="Number of try-on variations to generate",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Virtual Try-On: See how you'd look in any clothing.

    Upload a photo of a clothing item (from any online store, catalog, or closet)
    and the system will generate a photorealistic image of the registered user
    wearing that exact outfit — preserving their face, body type, and all
    distinguishing features.

    **How it works:**
    1. Takes the user's stored reference photos (face + body)
    2. Takes the uploaded clothing image
    3. Generates a new image: same person, new clothes

    **Tips for best results:**
    - Use clear, well-lit clothing photos (product photos from stores work great)
    - Front-facing clothing shots work better than angled ones
    - The system works with any clothing: shirts, dresses, suits, jackets, etc.
    - Add a description for better results (e.g., "red silk evening gown")

    **Example use cases:**
    - Online shopping: "How would I look in this dress?"
    - Outfit planning: "Does this suit match my look?"
    - Fashion design: Test designs on real body types
    """
    # ── Validate user ────────────────────────────────────────────────
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found. Please register first.",
        )

    # ── Validate clothing image ──────────────────────────────────────
    if not clothing_image.content_type or not clothing_image.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image (JPEG, PNG, etc.)",
        )

    clothing_bytes = await clothing_image.read()
    if len(clothing_bytes) < 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Clothing image appears too small or corrupt.",
        )

    # ── Load person reference images ─────────────────────────────────
    reference_keys = user.reference_image_keys or []
    if not reference_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No reference images found for user. Please upload reference photos first.",
        )

    # Primary reference
    try:
        person_image = await storage.download_image(reference_keys[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load person reference image: {str(e)}",
        )

    # Additional references for stronger face anchoring
    additional_refs = []
    for key in reference_keys[1:]:
        try:
            img_bytes = await storage.download_image(key)
            additional_refs.append(img_bytes)
        except Exception:
            continue

    # ── Generate try-on ──────────────────────────────────────────────
    try:
        generated_images = await imagen_service.generate_tryon(
            person_image=person_image,
            clothing_image=clothing_bytes,
            additional_person_refs=additional_refs if additional_refs else None,
            clothing_description=clothing_description,
            aspect_ratio=aspect_ratio.value,
            number_of_images=number_of_images,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate try-on image: {str(e)}",
        )

    if not generated_images:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No try-on images were generated. Please try again with a different clothing image.",
        )

    # ── Store results ────────────────────────────────────────────────
    results = []

    for i, image_bytes in enumerate(generated_images):
        # Create DB record
        tryon_record = GeneratedTryon(
            user_id=user.id,
            prompt=clothing_description,
            clothing_image_key="",
            result_image_key="",
        )
        db.add(tryon_record)
        await db.flush()

        # Store clothing image
        clothing_key = f"tryons/{user.id}/{tryon_record.id}_clothing.jpg"
        await storage.upload_image(clothing_bytes, clothing_key, content_type="image/jpeg")

        # Store result image
        result_key = f"tryons/{user.id}/{tryon_record.id}_result.jpg"
        await storage.upload_image(image_bytes, result_key, content_type="image/jpeg")

        # Update record with storage keys
        tryon_record.clothing_image_key = clothing_key
        tryon_record.result_image_key = result_key

        clothing_url = await storage.get_url(clothing_key)
        result_url = await storage.get_url(result_key)

        tryon_record.clothing_image_url = clothing_url
        tryon_record.result_image_url = result_url

        results.append(
            GeneratedTryonInfo(
                id=tryon_record.id,
                clothing_image_url=clothing_url,
                result_image_url=result_url,
                prompt=clothing_description,
                created_at=tryon_record.created_at,
            )
        )

    await db.commit()

    return VirtualTryonResponse(
        user_id=user.id,
        clothing_description=clothing_description,
        results=results,
    )


@router.get("/history/{user_id}", response_model=list[GeneratedTryonInfo])
async def get_tryon_history(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Get the virtual try-on history for a user."""
    # Check user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )

    # Get try-on records
    result = await db.execute(
        select(GeneratedTryon)
        .where(GeneratedTryon.user_id == user_id)
        .order_by(GeneratedTryon.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    tryons = result.scalars().all()

    # Build response with URLs
    tryon_infos = []
    for t in tryons:
        if t.clothing_image_key:
            t.clothing_image_url = await storage.get_url(t.clothing_image_key)
        if t.result_image_key:
            t.result_image_url = await storage.get_url(t.result_image_key)
        tryon_infos.append(GeneratedTryonInfo.model_validate(t))

    return tryon_infos
