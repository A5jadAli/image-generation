import io
import json
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass, field
from insightface.app import FaceAnalysis

from app.config import get_settings

settings = get_settings()


@dataclass
class FacialAttributes:
    """Extracted facial attributes for prompt enhancement."""
    estimated_age: int | None = None
    gender: str | None = None  # "male" or "female"
    # Additional attributes that can help with generation
    has_glasses: bool = False
    has_beard: bool = False
    skin_tone: str | None = None  # "light", "medium", "dark"

    def to_dict(self) -> dict:
        return {
            "estimated_age": self.estimated_age,
            "gender": self.gender,
            "has_glasses": self.has_glasses,
            "has_beard": self.has_beard,
            "skin_tone": self.skin_tone,
        }

    def to_prompt_description(self) -> str:
        """Convert attributes to a text description for prompt enhancement."""
        parts = []

        if self.gender:
            parts.append(self.gender)

        if self.estimated_age:
            if self.estimated_age < 25:
                parts.append("young adult")
            elif self.estimated_age < 40:
                parts.append("adult")
            elif self.estimated_age < 60:
                parts.append("middle-aged")
            else:
                parts.append("senior")

        if self.has_beard:
            parts.append("with beard")

        if self.has_glasses:
            parts.append("wearing glasses")

        return " ".join(parts) if parts else ""


@dataclass
class UserReferenceImages:
    """Collection of reference images for a user."""
    face_crop: bytes  # Close-up face crop
    upper_body_crop: bytes | None = None  # Face + shoulders/upper body
    full_image: bytes | None = None  # Best full image

    # Quality metrics
    face_quality_score: float = 0.0
    detection_confidence: float = 0.0

    # Facial attributes
    attributes: FacialAttributes = field(default_factory=FacialAttributes)

    # Face embedding for identity (512-dimensional vector)
    face_embedding: list[float] | None = None


@dataclass
class FaceResult:
    """Result of face detection and cropping."""
    cropped_image: bytes
    quality_score: float
    detection_confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    # Raw face object for attribute extraction
    _face_obj: object = None
    _original_image: np.ndarray = None


class FaceDetectionService:
    def __init__(self):
        self._face_analyzer: FaceAnalysis | None = None

    @property
    def face_analyzer(self) -> FaceAnalysis:
        """Lazy load the face analyzer with all analysis modules."""
        if self._face_analyzer is None:
            # buffalo_l includes: detection, recognition, age, gender, landmark
            self._face_analyzer = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_analyzer

    def _bytes_to_cv2(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format (BGR)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def _cv2_to_bytes(self, img: np.ndarray, format: str = "JPEG", quality: int = 95) -> bytes:
        """Convert OpenCV image to bytes."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format, quality=quality)
        return buffer.getvalue()

    def _calculate_quality_score(self, face, img: np.ndarray) -> float:
        """
        Calculate a quality score for the detected face.
        Higher score = better quality face for reference.
        """
        score = 0.0

        # 1. Detection confidence (0-1)
        det_score = float(face.det_score)
        score += det_score * 30  # Max 30 points

        # 2. Face size relative to image
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        img_height, img_width = img.shape[:2]
        face_area_ratio = (face_width * face_height) / (img_width * img_height)
        size_score = min(face_area_ratio * 100, 25)  # Max 25 points
        score += size_score

        # 3. Face pose (prefer frontal faces)
        if hasattr(face, 'pose') and face.pose is not None:
            pose = face.pose
            yaw_penalty = abs(pose[1]) / 90 * 20
            pitch_penalty = abs(pose[0]) / 90 * 10
            score += 30 - yaw_penalty - pitch_penalty
        else:
            score += 15

        # 4. Sharpness (Laplacian variance)
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        face_region = img[y1:y2, x1:x2]
        if face_region.size > 0:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100, 15)
            score += sharpness_score

        return score

    def _extract_attributes(self, face) -> FacialAttributes:
        """Extract facial attributes from InsightFace detection."""
        attributes = FacialAttributes()

        # Age estimation
        if hasattr(face, 'age') and face.age is not None:
            attributes.estimated_age = int(face.age)

        # Gender detection
        if hasattr(face, 'gender') and face.gender is not None:
            # InsightFace returns 0 for female, 1 for male
            attributes.gender = "male" if face.gender == 1 else "female"

        # Note: InsightFace buffalo_l doesn't detect glasses/beard directly
        # We could add additional models for this, but keeping it simple for now

        return attributes

    def _crop_face_tight(self, img: np.ndarray, bbox: tuple, padding: float = 0.3) -> np.ndarray:
        """Crop face region with padding - tight crop for face reference."""
        x1, y1, x2, y2 = bbox
        img_height, img_width = img.shape[:2]

        face_width = x2 - x1
        face_height = y2 - y1

        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)

        new_x1 = max(0, x1 - pad_x)
        new_y1 = max(0, y1 - pad_y)
        new_x2 = min(img_width, x2 + pad_x)
        new_y2 = min(img_height, y2 + pad_y)

        return img[new_y1:new_y2, new_x1:new_x2]

    def _crop_upper_body(self, img: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Crop upper body region - includes face, neck, and shoulders.
        This provides more context for body proportions.
        """
        x1, y1, x2, y2 = bbox
        img_height, img_width = img.shape[:2]

        face_width = x2 - x1
        face_height = y2 - y1
        face_center_x = (x1 + x2) // 2

        # Upper body crop: wider and extends below face
        # Width: 2.5x face width (to include shoulders)
        # Height: from above head to below shoulders (about 2.5x face height below)
        body_width = int(face_width * 2.5)
        body_height_above = int(face_height * 0.5)  # Above head
        body_height_below = int(face_height * 2.0)  # Below chin (shoulders)

        new_x1 = max(0, face_center_x - body_width // 2)
        new_x2 = min(img_width, face_center_x + body_width // 2)
        new_y1 = max(0, y1 - body_height_above)
        new_y2 = min(img_height, y2 + body_height_below)

        return img[new_y1:new_y2, new_x1:new_x2]

    def _detect_skin_tone(self, img: np.ndarray, bbox: tuple) -> str:
        """Estimate skin tone from face region."""
        x1, y1, x2, y2 = bbox

        # Get center of face (avoiding hair/background)
        center_x1 = x1 + (x2 - x1) // 4
        center_x2 = x2 - (x2 - x1) // 4
        center_y1 = y1 + (y2 - y1) // 4
        center_y2 = y2 - (y2 - y1) // 4

        face_center = img[center_y1:center_y2, center_x1:center_x2]

        if face_center.size == 0:
            return "medium"

        # Convert to LAB color space for better skin tone analysis
        lab = cv2.cvtColor(face_center, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        avg_lightness = np.mean(l_channel)

        # Classify based on lightness
        if avg_lightness > 180:
            return "light"
        elif avg_lightness > 120:
            return "medium"
        else:
            return "dark"

    def detect_faces_with_details(self, image_bytes: bytes) -> list[FaceResult]:
        """Detect faces and return detailed results including raw face object."""
        img = self._bytes_to_cv2(image_bytes)
        faces = self.face_analyzer.get(img)

        results = []
        for face in faces:
            bbox = tuple(face.bbox.astype(int))

            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            if face_width < settings.min_face_size or face_height < settings.min_face_size:
                continue

            quality_score = self._calculate_quality_score(face, img)

            cropped_img = self._crop_face_tight(img, bbox, padding=settings.face_padding)
            cropped_bytes = self._cv2_to_bytes(cropped_img)

            results.append(FaceResult(
                cropped_image=cropped_bytes,
                quality_score=quality_score,
                detection_confidence=float(face.det_score),
                bbox=bbox,
                _face_obj=face,
                _original_image=img,
            ))

        return results

    def process_user_images(self, images: list[bytes]) -> UserReferenceImages | None:
        """
        Process multiple user images and extract:
        1. Best face crop (tight)
        2. Best upper body crop
        3. Best full image
        4. Facial attributes
        5. Face embedding
        """
        all_results: list[tuple[FaceResult, bytes]] = []  # (result, original_image_bytes)

        for image_bytes in images:
            faces = self.detect_faces_with_details(image_bytes)
            for face_result in faces:
                all_results.append((face_result, image_bytes))

        if not all_results:
            return None

        # Sort by quality score
        all_results.sort(key=lambda x: x[0].quality_score, reverse=True)

        best_result, best_original_bytes = all_results[0]
        best_face = best_result._face_obj
        best_img = best_result._original_image

        # 1. Face crop (already done)
        face_crop = best_result.cropped_image

        # 2. Upper body crop
        upper_body_img = self._crop_upper_body(best_img, best_result.bbox)
        upper_body_crop = self._cv2_to_bytes(upper_body_img)

        # 3. Best full image (resize if too large, keep aspect ratio)
        full_img = best_img.copy()
        max_dim = 1024
        h, w = full_img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            full_img = cv2.resize(full_img, (int(w * scale), int(h * scale)))
        full_image = self._cv2_to_bytes(full_img)

        # 4. Extract attributes
        attributes = self._extract_attributes(best_face)
        attributes.skin_tone = self._detect_skin_tone(best_img, best_result.bbox)

        # 5. Face embedding (for potential future use - identity matching)
        face_embedding = None
        if hasattr(best_face, 'embedding') and best_face.embedding is not None:
            face_embedding = best_face.embedding.tolist()

        return UserReferenceImages(
            face_crop=face_crop,
            upper_body_crop=upper_body_crop,
            full_image=full_image,
            face_quality_score=best_result.quality_score,
            detection_confidence=best_result.detection_confidence,
            attributes=attributes,
            face_embedding=face_embedding,
        )

    # Keep old method for backward compatibility
    def get_best_face_from_images(self, images: list[bytes]) -> FaceResult | None:
        """Process multiple images and return the best quality face."""
        all_faces: list[FaceResult] = []

        for image_bytes in images:
            faces = self.detect_faces_with_details(image_bytes)
            all_faces.extend(faces)

        if not all_faces:
            return None

        all_faces.sort(key=lambda x: x.quality_score, reverse=True)
        return all_faces[0]


# Singleton instance
face_detection_service = FaceDetectionService()
