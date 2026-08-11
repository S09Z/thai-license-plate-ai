"""Application configuration loaded from the environment."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings read from environment variables (prefix ``APP_``).

    Attributes:
        app_name: Human-readable service name.
        env: Deployment environment (``dev``, ``staging`` or ``prod``).
        log_level: Root logging level (e.g. ``INFO``, ``DEBUG``).
        version: Application version, mirrored from ``pyproject.toml``.
        detector_model_path: Path to the YOLO plate-detection weights.
        detector_conf_threshold: Minimum confidence for a kept detection.
        face_model_path: Path to the YuNet ONNX face-detection model.
        face_conf_threshold: Minimum score for a reported face.
        face_landmark_model_path: Path to the OpenCV LBF 68-point model.
        face_fast_max_size: Longest-edge cap for the fast face-detection path;
            larger inputs are downscaled server-side before inference.
        face_gender_model_path: Path to the Levi-Hassner Caffe gender weights.
        face_gender_proto_path: Path to the gender network's Caffe prototxt.
        face_expression_model_path: Path to the ONNX facial-expression model.
        face_attribute_min_confidence: Score below which an inferred expression
            or apparent gender is reported as ``None`` rather than guessed.
        max_upload_bytes: Largest accepted image upload, in bytes.
        allowed_image_types: Content types accepted by image endpoints.
        ocr_lang: PaddleOCR language code used for recognition.
        ocr_min_confidence: Recognized lines below this score are discarded.
        ocr_det_limit_side_len: Longest edge, in pixels, PaddleOCR's text
            detector resizes the crop to before detection. The detector (DBNet)
            dominates OCR latency; capping this below the crop's 256px width
            downscales the detector's input and cuts its cost.
        ocr_det_limit_type: Which edge ``ocr_det_limit_side_len`` bounds;
            ``"max"`` caps the longest edge so the small crop is downscaled.
        plate_crop_width: Width of the rectified plate crop handed to OCR.
        plate_crop_height: Height of the rectified plate crop handed to OCR.
        plate_crop_padding: Fraction of box size the crop is expanded by.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "thai-license-plate-ai"
    env: str = "dev"
    log_level: str = "INFO"
    version: str = "0.1.0"

    detector_model_path: str = "models/detector/best.pt"
    detector_conf_threshold: float = 0.25
    face_model_path: str = "models/face/face_detection_yunet_2023mar.onnx"
    face_conf_threshold: float = 0.6
    face_landmark_model_path: str = "models/face/lbfmodel.yaml"
    # Roughly a third of a 720p edge. At this size YuNet costs a few ms rather
    # than ~17 ms, which is what lets the realtime loop run at camera cadence.
    face_fast_max_size: int = 480
    face_gender_model_path: str = "models/face/gender_net.caffemodel"
    face_gender_proto_path: str = "models/face/gender_deploy.prototxt"
    face_expression_model_path: str = "models/face/expression.onnx"
    face_attribute_min_confidence: float = 0.5
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png")

    ocr_lang: str = "th"
    ocr_min_confidence: float = 0.5
    # PaddleOCR's DBNet text detector is ~90% of OCR latency and runs on the
    # crop's long edge. Capping it at 192px (below the 256px crop width) cut OCR
    # ~28% with no change to plate/CER/province accuracy on the real-plate eval
    # set (Phase 15b, docs/benchmark/recognize-latency-phase15b.md).
    ocr_det_limit_side_len: int = 192
    ocr_det_limit_type: str = "max"

    # A placeholder ratio, not a measured Thai-plate one. Exposed here so it
    # can be calibrated against real photographs without a code change.
    plate_crop_width: int = 256
    plate_crop_height: int = 128
    plate_crop_padding: float = 0.04


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Returns:
        The process-wide settings, instantiated once and memoized.
    """
    return Settings()
