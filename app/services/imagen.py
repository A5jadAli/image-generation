import io
from google import genai
from google.genai import types
from PIL import Image
from app.config import get_settings

settings = get_settings()


class NanoBananaService:
    """
    Service for generating images using Google's Nano Banana (Gemini Image Generation) API.

    Supports:
    1. Personalized image generation — preserves person's face from reference
    2. Virtual try-on — puts person into uploaded clothing
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

    # ─── Scene & mood enhancement ────────────────────────────────────────

    def _enhance_user_prompt(self, user_prompt: str) -> str:
        """Add scene and mood details to a short user prompt."""
        prompt_lower = user_prompt.lower()

        scene_enhancements = {
            "beach": "warm natural sunlight, ocean in background, sand texture, casual atmosphere, candid realism",
            "gym": "overhead gym lights, worn equipment in background, natural sweat, candid phone photo feel",
            "cafe": "indoor ambient light, coffee shop background, casual seating, everyday realism",
            "coffee": "warm indoor light, cafe background, relaxed atmosphere, candid moment",
            "office": "fluorescent office lighting, real workspace clutter, natural daylight from windows, casual realism",
            "nature": "natural outdoor light, real greenery, unposed outdoor moment, candid realism",
            "city": "urban street setting, natural city light, pedestrians in background, shot like a casual phone photo",
            "home": "warm indoor lighting, lived-in interior, everyday home setting, candid realism",
            "party": "mixed indoor lighting, real party setting, candid moment capture, not staged",
            "wedding": "soft natural light, real venue setting, elegant but candid, genuine moment",
            "mountain": "natural outdoor light, real mountain backdrop, hiking gear, candid trail moment",
            "restaurant": "warm restaurant lighting, real table setting, casual dining moment, candid realism",
            "park": "natural daylight, real park setting, trees and grass, casual outdoor moment",
            "hiking": "natural trail lighting, outdoor scenery, real hiking moment, candid phone photo feel",
            "graduation": "outdoor ceremony light, real academic setting, proud but candid moment",
            "concert": "stage lights and crowd, live event atmosphere, candid fan photo energy",
        }

        mood_mappings = {
            ("happy", "joy", "celebrating", "party", "fun", "laugh", "smile"):
                "natural smile, relaxed and candid, not overly posed",
            ("confident", "business", "presentation", "professional", "meeting"):
                "natural confident look, casual composure, realistic body language",
            ("relaxed", "beach", "vacation", "spa", "peaceful", "calm"):
                "relaxed natural expression, casual unposed body language",
            ("excited", "adventure", "travel", "sports", "thrilled"):
                "natural excited expression, caught mid-moment, candid energy",
            ("romantic", "date", "wedding", "love", "dinner"):
                "soft natural expression, genuine warmth, candid moment",
            ("elegant", "gala", "formal", "luxury", "fashion"):
                "natural poise, real-feeling elegance, not overly staged",
            ("thoughtful", "reading", "studying", "working", "thinking"):
                "naturally focused, candid mid-thought moment, realistic gaze",
        }

        parts = [user_prompt]

        for scene, enhancement in scene_enhancements.items():
            if scene in prompt_lower:
                parts.append(enhancement)
                break

        expression_added = False
        for keywords, expression in mood_mappings.items():
            if any(kw in prompt_lower for kw in keywords):
                parts.append(expression)
                expression_added = True
                break

        if not expression_added:
            parts.append("natural expression, candid and unposed")

        return ", ".join(parts)

    # ─── Prompt builders ─────────────────────────────────────────────────

    def _build_generation_prompt(self, user_prompt: str) -> str:
        """
        Build a concise, high-signal prompt for identity-preserving generation.

        Key insight: Gemini responds better to SHORT, DIRECT identity instructions
        than to walls of text. The reference image does most of the work — the prompt
        just needs to anchor the model to it firmly.
        """
        enhanced_scene = self._enhance_user_prompt(user_prompt)

        prompt = (
            f"This is a photo of a specific person. Generate a new image "
            f"of THIS EXACT SAME PERSON in the following scene: {enhanced_scene}.\n\n"
            f"CRITICAL RULES:\n"
            f"1. The person's face must be IDENTICAL to the reference — same bone structure, "
            f"same eyes, same nose, same lips, same jawline, same skin tone. Do NOT alter "
            f"any facial feature. The person must be instantly recognizable.\n"
            f"2. Preserve ALL distinguishing features: glasses, facial hair, moles, freckles, "
            f"scars, birthmarks, hair color, hair texture, ear shape, eyebrow shape.\n"
            f"3. Keep the same body type and proportions as the reference.\n"
            f"4. Any expression change must only involve natural muscle movement — the underlying "
            f"facial geometry must remain unchanged.\n"
            f"5. Natural realistic lighting for the scene, realistic skin texture with natural "
            f"imperfections, candid phone photo quality. Avoid overly polished or studio-perfect "
            f"look. The image should feel like a real candid photo, not AI-generated. Almost real.\n\n"
            f"Scene: {user_prompt}"
        )

        return prompt

    def _build_tryon_prompt(self, clothing_description: str = "") -> str:
        """
        Build prompt for virtual try-on.

        Strategy: Two images are sent — Image 1 (person) and Image 2 (clothing).
        The prompt instructs the model to dress the person in the exact garment.
        """
        base_prompt = (
            "I am providing two images:\n"
            "- IMAGE 1 (first image): A photo of a specific person. This is the PERSON reference.\n"
            "- IMAGE 2 (second image): A photo of a clothing item/outfit. This is the CLOTHING reference.\n\n"
            "TASK: Generate a new photorealistic image showing the EXACT SAME PERSON from Image 1 "
            "wearing the EXACT SAME CLOTHING from Image 2.\n\n"
            "PERSON IDENTITY RULES (NON-NEGOTIABLE):\n"
            "- The face must be IDENTICAL to Image 1 — same bone structure, eyes, nose, lips, "
            "jawline, skin tone, skin texture. NOT similar — IDENTICAL.\n"
            "- Preserve ALL distinguishing features: glasses, facial hair, moles, freckles, "
            "birthmarks, hair color, hair texture, ear shape, eyebrow shape.\n"
            "- Same body type and proportions as Image 1.\n\n"
            "CLOTHING RULES (NON-NEGOTIABLE):\n"
            "- The clothing must match Image 2 EXACTLY — same design, same color, same pattern, "
            "same fabric texture, same style details (buttons, zippers, collars, prints, logos).\n"
            "- The clothing should fit naturally on the person's body — proper draping, "
            "realistic wrinkles and folds based on their body shape.\n"
            "- Do NOT modify the clothing design in any way. Do NOT change the color or pattern.\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "- Full body or upper body shot showing the clothing clearly.\n"
            "- Professional photography quality, studio or lifestyle setting.\n"
            "- Natural pose that shows off the clothing well.\n"
            "- Clean, well-lit background that doesn't distract from the outfit.\n"
            "- Photorealistic result — should look like a real photo, not AI-generated."
        )

        if clothing_description:
            base_prompt += (
                f"\n\nADDITIONAL CONTEXT: {clothing_description}"
            )

        return base_prompt

    # ─── Image extraction helper ─────────────────────────────────────────

    def _extract_images_from_response(self, response) -> list[bytes]:
        """Extract generated image bytes from a Gemini API response."""
        images = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data and part.inline_data.data:
                            images.append(part.inline_data.data)
        return images

    # ─── Core generation methods ─────────────────────────────────────────

    async def generate_image(
        self,
        prompt: str,
        reference_image: bytes,
        additional_references: list[bytes] | None = None,
        aspect_ratio: str = "1:1",
        number_of_images: int = 1,
    ) -> list[bytes]:
        """
        Generate images preserving the person's identity from reference.

        Sends multiple reference images when available for stronger face anchoring.
        """
        full_prompt = self._build_generation_prompt(user_prompt=prompt)

        # Convert reference images to PIL
        reference_pil = self._bytes_to_pil_image(reference_image)

        # Build contents: prompt first, then all reference images
        # Sending multiple refs gives the model more angles to lock onto the face
        contents = [full_prompt, reference_pil]

        if additional_references:
            for ref_bytes in additional_references[:3]:  # Max 3 extra refs
                try:
                    ref_pil = self._bytes_to_pil_image(ref_bytes)
                    contents.append(ref_pil)
                except Exception:
                    continue

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

            generated_images.extend(self._extract_images_from_response(response))

        return generated_images

    async def generate_tryon(
        self,
        person_image: bytes,
        clothing_image: bytes,
        additional_person_refs: list[bytes] | None = None,
        clothing_description: str = "",
        aspect_ratio: str = "3:4",
        number_of_images: int = 1,
    ) -> list[bytes]:
        """
        Virtual try-on: Generate image of person wearing specific clothing.

        Args:
            person_image: Primary reference image of the person
            clothing_image: Image of the clothing item to try on
            additional_person_refs: Extra person photos for stronger face anchoring
            clothing_description: Optional text describing the clothing
            aspect_ratio: Output aspect ratio (default 3:4 for fashion)
            number_of_images: Number of variations to generate (1-4)

        Returns:
            List of generated try-on image bytes
        """
        prompt = self._build_tryon_prompt(clothing_description)

        # Convert images to PIL
        person_pil = self._bytes_to_pil_image(person_image)
        clothing_pil = self._bytes_to_pil_image(clothing_image)

        # Build contents: prompt → person image(s) → clothing image
        # Order matters: person refs first, clothing last, so the model
        # clearly distinguishes "who" from "what to wear"
        contents = [prompt, person_pil]

        # Add extra person references for stronger identity lock
        if additional_person_refs:
            for ref_bytes in additional_person_refs[:2]:  # Max 2 extra
                try:
                    ref_pil = self._bytes_to_pil_image(ref_bytes)
                    contents.append(ref_pil)
                except Exception:
                    continue

        # Clothing image goes last
        contents.append(clothing_pil)

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

            generated_images.extend(self._extract_images_from_response(response))

        return generated_images


# Singleton instance
imagen_service = NanoBananaService()
