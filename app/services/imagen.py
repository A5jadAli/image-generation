import base64
import httpx
from dataclasses import dataclass
from app.config import get_settings

settings = get_settings()


@dataclass
class ReferenceImages:
    """Collection of reference images for generation."""
    face_image: bytes
    upper_body_image: bytes | None = None
    full_image: bytes | None = None


class NanoBananaService:
    """
    Service for generating images using Google's Nano Banana (Gemini Image Generation) API.

    Nano Banana supports up to 5 reference images for person/subject generation,
    allowing us to maintain facial features and body characteristics when generating new images.
    """

    def __init__(self):
        self.api_key = settings.nano_banana_api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        # Use gemini-2.0-flash-exp for speed, or gemini-2.0-flash-preview-image-generation
        self.model = "gemini-2.0-flash-exp"

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    def _base64_to_image(self, base64_string: str) -> bytes:
        """Convert base64 string to image bytes."""
        return base64.b64decode(base64_string)

    def _build_prompt_with_attributes(
        self,
        base_prompt: str,
        person_description: str | None = None,
        negative_prompt: str | None = None,
    ) -> str:
        """
        Build an enhanced prompt with person attributes.

        Args:
            base_prompt: The user's scene description
            person_description: Description from facial attributes (e.g., "male adult")
            negative_prompt: Things to avoid
        """
        parts = []

        # Start with instruction to use reference images
        parts.append("Generate a realistic photo of the person shown in the reference images")

        # Add person description if available
        if person_description:
            parts.append(f"(a {person_description})")

        # Add the scene/activity
        parts.append(f"in this scenario: {base_prompt}")

        # Add quality instructions
        parts.append("Make sure to preserve the exact facial features, skin tone, and overall appearance from the reference images.")
        parts.append("High quality, photorealistic, natural lighting.")

        # Add negative prompt
        if negative_prompt:
            parts.append(f"Avoid: {negative_prompt}")

        return " ".join(parts)

    async def generate_image_with_references(
        self,
        prompt: str,
        reference_images: ReferenceImages,
        person_description: str | None = None,
        aspect_ratio: str = "1:1",
        number_of_images: int = 1,
        negative_prompt: str | None = None,
    ) -> list[bytes]:
        """
        Generate images using Nano Banana with multiple reference images.

        Sends face crop, upper body crop, and full image as references
        to maintain consistent appearance.

        Args:
            prompt: Text description of the scene/activity
            reference_images: Collection of reference images (face, upper body, full)
            person_description: Description from facial attributes for prompt enhancement
            aspect_ratio: Output image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
            number_of_images: Number of images to generate (1-4)
            negative_prompt: What to avoid in the generated image

        Returns:
            List of generated image bytes
        """
        # Build enhanced prompt
        full_prompt = self._build_prompt_with_attributes(
            base_prompt=prompt,
            person_description=person_description,
            negative_prompt=negative_prompt,
        )

        # Build parts array with text and multiple reference images
        parts = [{"text": full_prompt}]

        # Add face image (always included - most important)
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": self._image_to_base64(reference_images.face_image)
            }
        })

        # Add upper body image if available
        if reference_images.upper_body_image:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": self._image_to_base64(reference_images.upper_body_image)
                }
            })

        # Add full image if available
        if reference_images.full_image:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": self._image_to_base64(reference_images.full_image)
                }
            })

        # Build the request payload
        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "candidateCount": number_of_images,
            }
        }

        # Make the API request
        url = f"{self.base_url}/{self.model}:generateContent"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                }
            )

            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"Nano Banana API error ({response.status_code}): {error_detail}")

            result = response.json()

        # Extract generated images from response
        generated_images = []

        if "candidates" in result:
            for candidate in result["candidates"]:
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "inlineData" in part:
                            image_base64 = part["inlineData"]["data"]
                            image_bytes = self._base64_to_image(image_base64)
                            generated_images.append(image_bytes)

        return generated_images

    # Keep backward compatible method
    async def generate_image_with_reference(
        self,
        prompt: str,
        reference_face_image: bytes,
        aspect_ratio: str = "1:1",
        number_of_images: int = 1,
        negative_prompt: str | None = None,
    ) -> list[bytes]:
        """
        Generate images using a single reference face image.
        Backward compatible method - use generate_image_with_references for better results.
        """
        return await self.generate_image_with_references(
            prompt=prompt,
            reference_images=ReferenceImages(face_image=reference_face_image),
            aspect_ratio=aspect_ratio,
            number_of_images=number_of_images,
            negative_prompt=negative_prompt,
        )

    async def generate_image_simple(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        number_of_images: int = 1,
        negative_prompt: str | None = None,
    ) -> list[bytes]:
        """
        Generate images without a reference (simple text-to-image).
        """
        full_prompt = prompt
        if negative_prompt:
            full_prompt += f". Avoid: {negative_prompt}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "candidateCount": number_of_images,
            }
        }

        url = f"{self.base_url}/{self.model}:generateContent"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                }
            )

            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"Nano Banana API error ({response.status_code}): {error_detail}")

            result = response.json()

        generated_images = []

        if "candidates" in result:
            for candidate in result["candidates"]:
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "inlineData" in part:
                            image_base64 = part["inlineData"]["data"]
                            image_bytes = self._base64_to_image(image_base64)
                            generated_images.append(image_bytes)

        return generated_images


# Singleton instance
imagen_service = NanoBananaService()
