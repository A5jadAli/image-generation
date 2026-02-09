import io
import asyncio
import imghdr
from google import genai
from google.genai import types

from app.config import get_settings

settings = get_settings()

# Map imghdr results to MIME types
MIME_TYPE_MAP = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
}


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect the MIME type of image bytes."""
    img_type = imghdr.what(None, h=image_bytes)
    return MIME_TYPE_MAP.get(img_type, "image/jpeg")


def _bytes_to_genai_image(image_bytes: bytes) -> types.Image:
    """Convert raw image bytes to a Google GenAI Image object with proper mime type."""
    mime_type = _detect_mime_type(image_bytes)
    return types.Image(image_bytes=image_bytes, mime_type=mime_type)


class VeoVideoService:
    """
    Service for generating videos using Google's Veo 3.1 API.

    Supports two modes:
    1. Text-to-video with optional reference images for personalization
    2. Image-to-video from an existing generated image
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.nano_banana_api_key)
        self.video_model = "veo-3.1-generate-preview"
        self.poll_interval = 10  # seconds

    def _enhance_video_prompt(self, user_prompt: str) -> str:
        """
        Enhance a user prompt with cinematic and video-specific details.
        """
        prompt_lower = user_prompt.lower()

        # Scene/movement enhancements for video
        scene_enhancements = {
            "beach": "gentle ocean waves, soft breeze movement, golden hour lighting",
            "walking": "smooth walking motion, natural gait, subtle environment movement",
            "dancing": "fluid dance movements, rhythmic motion, dynamic energy",
            "talking": "natural lip movements, expressive gestures, conversational tone",
            "sitting": "subtle breathing motion, relaxed posture, ambient environment",
            "standing": "confident stance, subtle natural movements, professional presence",
            "running": "dynamic running motion, athletic movement, energetic pace",
            "cooking": "hands in motion, kitchen activity, steam and movement",
            "working": "typing or writing motions, focused activity, office ambiance",
            "nature": "wind through trees, moving clouds, natural ambient motion",
            "city": "urban activity, passing traffic, city life movement",
        }

        enhanced_parts = [user_prompt]

        # Add scene enhancements
        for scene, enhancement in scene_enhancements.items():
            if scene in prompt_lower:
                enhanced_parts.append(enhancement)
                break

        # Add cinematic quality
        enhanced_parts.append("cinematic quality, smooth motion, professional lighting")

        return ", ".join(enhanced_parts)

    def _build_video_prompt(
        self, user_prompt: str, for_personalization: bool = False
    ) -> str:
        """
        Build a concise prompt optimized for video generation.
        Veo 3.1 works best with short, descriptive prompts. Overly verbose
        or instructional prompts tend to trigger audio safety filters.
        """
        enhanced_prompt = self._enhance_video_prompt(user_prompt)

        if for_personalization:
            prompt = (
                f"Cinematic video of the person from the reference images: "
                f"{enhanced_prompt}. Photorealistic, consistent identity throughout."
            )
        else:
            prompt = (
                f"Cinematic video: {enhanced_prompt}. Photorealistic, high fidelity."
            )

        return prompt

    async def _poll_operation(self, operation) -> bytes:
        """
        Poll the operation until video generation is complete.
        Returns the video bytes.
        """
        while not operation.done:
            await asyncio.sleep(self.poll_interval)
            operation = self.client.operations.get(operation)

        # Check for errors in the operation
        if hasattr(operation, "error") and operation.error:
            raise Exception(f"Video generation failed: {operation.error}")

        if not operation.response:
            # Log the full operation for debugging
            raise Exception(
                f"Video generation failed: No response. Operation: {operation}"
            )

        if not operation.response.generated_videos:
            raise Exception(
                f"Video generation failed: No videos generated. Response: {operation.response}"
            )

        # Get the first generated video
        video = operation.response.generated_videos[0]

        # Download the video
        self.client.files.download(file=video.video)

        # Read the video bytes
        video_bytes = video.video.read()

        return video_bytes

    async def generate_video_from_text(
        self,
        prompt: str,
        reference_images: list[bytes] | None = None,
    ) -> bytes:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Description of the video scene/action
            reference_images: Optional list of reference image bytes for personalization

        Returns:
            Video bytes (MP4 format)
        """
        # Build the prompt
        has_references = reference_images is not None and len(reference_images) > 0
        full_prompt = self._build_video_prompt(
            prompt, for_personalization=has_references
        )

        if has_references:
            # Create reference image objects with proper types.Image format
            reference_image_objects = []
            for img_bytes in reference_images:
                genai_image = _bytes_to_genai_image(img_bytes)
                ref_image = types.VideoGenerationReferenceImage(
                    image=genai_image, reference_type="asset"
                )
                reference_image_objects.append(ref_image)

            # Generate video with reference images
            operation = self.client.models.generate_videos(
                model=self.video_model,
                prompt=full_prompt,
                config=types.GenerateVideosConfig(
                    reference_images=reference_image_objects,
                ),
            )
        else:
            # Generate video without reference images
            operation = self.client.models.generate_videos(
                model=self.video_model,
                prompt=full_prompt,
            )

        # Poll until complete and return video bytes
        return await self._poll_operation(operation)

    async def generate_video_from_image(
        self,
        prompt: str,
        source_image: bytes,
    ) -> bytes:
        """
        Generate a video from an existing image (image-to-video).

        Args:
            prompt: Description of the motion/action for the video
            source_image: The source image bytes to animate

        Returns:
            Video bytes (MP4 format)
        """
        # Convert to Google GenAI Image with proper mime type
        genai_image = _bytes_to_genai_image(source_image)

        # Build a concise prompt for image-to-video
        enhanced_prompt = self._enhance_video_prompt(prompt)
        full_prompt = f"Animate this image: {enhanced_prompt}. Preserve appearance, smooth natural motion."

        # Generate video from the image
        operation = self.client.models.generate_videos(
            model=self.video_model,
            prompt=full_prompt,
            image=genai_image,
        )

        # Poll until complete and return video bytes
        return await self._poll_operation(operation)


# Singleton instance
video_service = VeoVideoService()
