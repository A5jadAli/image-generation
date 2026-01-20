import io
from google import genai
from google.genai import types
from PIL import Image
from app.config import get_settings

settings = get_settings()


class NanoBananaService:
    """
    Service for generating images using Google's Nano Banana (Gemini Image Generation) API.

    Uses a single reference image to maintain facial features when generating new images.
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.nano_banana_api_key)
        self.model = "gemini-3-pro-image-preview"

    def _bytes_to_pil_image(self, image_bytes: bytes) -> Image.Image:
        """Convert image bytes to PIL Image."""
        return Image.open(io.BytesIO(image_bytes))

    def _pil_image_to_bytes(self, pil_image: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        return buffer.read()

    def _build_generation_prompt(self, user_prompt: str) -> str:
        """
        Build a strict prompt that instructs Nano Banana to generate an image
        while strictly preserving the person's facial features from the reference.
        """
        prompt = f"""You are an AI image generator specialized in creating images of a SPECIFIC person while maintaining their EXACT identity.

TASK: Generate a new image of the EXACT SAME person shown in the reference image, depicting them in this scenario: {user_prompt}

CRITICAL IDENTITY PRESERVATION RULES - YOU MUST FOLLOW ALL OF THESE:

1. FACE: The generated face MUST be identical to the reference image:
   - Same exact facial structure and bone structure
   - Same eye shape, eye color, eye spacing
   - Same nose shape and size
   - Same lip shape and mouth structure
   - Same eyebrow shape and thickness
   - Same jawline and chin shape
   - Same forehead shape and hairline
   - Same ear shape if visible

2. SKIN: Preserve exact skin characteristics:
   - Same skin tone and complexion
   - Same skin texture
   - Any visible moles, freckles, or birthmarks

3. HAIR: Keep hair characteristics consistent:
   - Same hair color
   - Same hair texture (straight, wavy, curly)
   - Similar hairstyle (can be slightly different due to scene context)

4. BODY: Maintain body characteristics:
   - Same approximate body type/build
   - Same proportions

5. DISTINGUISHING FEATURES: Preserve ALL unique identifying features:
   - Facial hair if present (beard, mustache)
   - Glasses if worn in reference
   - Any visible scars or unique marks

OUTPUT REQUIREMENTS:
- Photorealistic, high-quality image
- Natural lighting appropriate for the scene
- The person should be clearly recognizable as the SAME individual from the reference
- Only change the setting/scene/activity/clothing as needed for: {user_prompt}

DO NOT:
- Change any facial features
- Alter skin tone or complexion
- Generate a different person
- Modify the person's apparent age significantly
- Change eye color or facial structure"""

        return prompt

    async def generate_image(
        self,
        prompt: str,
        reference_image: bytes,
        aspect_ratio: str = "1:1",
        number_of_images: int = 1,
    ) -> list[bytes]:
        """
        Generate images using Nano Banana with a reference image.

        The reference image is sent along with a strict prompt that instructs
        the model to preserve the person's facial features exactly.

        Args:
            prompt: User's description of the scene/activity
            reference_image: The reference image bytes (original uploaded image)
            aspect_ratio: Output aspect ratio
            number_of_images: Number of images to generate (1-4)

        Returns:
            List of generated image bytes
        """
        # Build the strict prompt with identity preservation instructions
        full_prompt = self._build_generation_prompt(user_prompt=prompt)

        # Convert reference image bytes to PIL Image
        reference_pil_image = self._bytes_to_pil_image(reference_image)

        # Build contents list with prompt and reference image
        contents = [full_prompt, reference_pil_image]

        # Generate images using the SDK
        generated_images = []

        for _ in range(number_of_images):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                ),
            )

            # Extract generated images from response
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.inline_data and part.inline_data.data:
                                generated_images.append(part.inline_data.data)

        return generated_images


# Singleton instance
imagen_service = NanoBananaService()
