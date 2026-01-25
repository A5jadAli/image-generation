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

    def _enhance_user_prompt(self, user_prompt: str) -> str:
        """
        Enhance a short user prompt with scene details and quality keywords.
        """
        prompt_lower = user_prompt.lower()

        # Scene enhancement mappings
        scene_enhancements = {
            "beach": "golden hour sunlight, ocean waves, warm tropical tones",
            "gym": "modern fitness equipment, motivational atmosphere, dynamic energy",
            "cafe": "cozy warm ambiance, artistic interior, soft natural light",
            "coffee": "cozy cafe setting, warm atmosphere, comfortable seating",
            "office": "modern professional workspace, clean aesthetic, natural daylight",
            "nature": "lush greenery, vibrant colors, fresh outdoor atmosphere",
            "city": "urban street photography, architectural backdrop, city life",
            "home": "comfortable interior, warm homely atmosphere, soft lighting",
            "party": "festive decorations, celebratory atmosphere, vibrant energy",
            "wedding": "elegant romantic setting, beautiful decorations, soft dreamy light",
            "mountain": "majestic peaks, breathtaking scenery, adventure spirit",
            "restaurant": "fine dining ambiance, elegant table setting, warm lighting",
            "park": "green surroundings, peaceful environment, natural beauty",
            "hiking": "scenic trail, outdoor adventure, natural landscape",
            "graduation": "academic ceremony, proud achievement, formal setting",
            "concert": "dynamic stage lighting, energetic atmosphere, live music vibes",
        }

        # Mood/expression mappings
        mood_mappings = {
            ("happy", "joy", "celebrating", "party", "fun", "laugh", "smile"):
                "genuine warm smile, joyful sparkling eyes, radiating happiness",
            ("confident", "business", "presentation", "professional", "meeting"):
                "confident composed expression, self-assured posture, professional demeanor",
            ("relaxed", "beach", "vacation", "spa", "peaceful", "calm"):
                "serene relaxed expression, peaceful calm mood, comfortable natural pose",
            ("excited", "adventure", "travel", "sports", "thrilled"):
                "bright enthusiastic expression, excited eyes, dynamic energy",
            ("romantic", "date", "wedding", "love", "dinner"):
                "soft tender expression, warm loving gaze, romantic mood",
            ("elegant", "gala", "formal", "luxury", "fashion"):
                "graceful refined expression, sophisticated poise, elegant demeanor",
            ("thoughtful", "reading", "studying", "working", "thinking"):
                "contemplative thoughtful look, focused intelligent gaze",
        }

        # Build enhanced prompt
        enhanced_parts = [user_prompt]

        # Add scene enhancements
        for scene, enhancement in scene_enhancements.items():
            if scene in prompt_lower:
                enhanced_parts.append(enhancement)
                break

        # Add mood/expression
        expression_added = False
        for keywords, expression in mood_mappings.items():
            if any(kw in prompt_lower for kw in keywords):
                enhanced_parts.append(expression)
                expression_added = True
                break

        if not expression_added:
            enhanced_parts.append("natural authentic expression, genuine emotion")

        return ", ".join(enhanced_parts)

    def _build_generation_prompt(self, user_prompt: str) -> str:
        """
        Build a prompt that generates beautiful, realistic images
        while preserving the person's facial features from the reference.
        """
        # Enhance the user's short prompt
        enhanced_scene = self._enhance_user_prompt(user_prompt)

        prompt = f"""Generate a stunning photorealistic image of the EXACT person from the reference photo in this scene: {enhanced_scene}

IDENTITY PRESERVATION (CRITICAL):
- Face MUST match reference exactly: same facial structure, eyes, nose, lips, jawline
- Same skin tone, texture, and any visible marks (moles, freckles)
- Same hair color and texture
- Same body type and proportions
- Preserve glasses, facial hair, or other distinguishing features if present

IMAGE QUALITY REQUIREMENTS:
- Professional photography quality, 8K ultra HD resolution
- Beautiful natural lighting that complements the scene
- Cinematic composition with artistic framing
- Rich vibrant colors, perfect exposure
- Sharp focus on the subject with pleasing depth of field
- Detailed realistic skin texture (not overly smooth or artificial)

EXPRESSION & POSE (MUST PRESERVE IDENTITY):
- Expressions must NOT alter underlying facial bone structure
- When smiling: same eye shape, same nose, same jawline - only natural muscle movement
- The face must remain instantly recognizable as the same person even with different expressions
- Natural authentic expression appropriate for the scene
- Eyes should look alive and expressive while keeping same eye shape and spacing

SCENE: {user_prompt}
- Create an immersive, believable environment
- Appropriate clothing and styling for the context
- Harmonious color palette between subject and background

The final image should look like a professional photograph taken by a skilled photographer, not AI-generated."""

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
