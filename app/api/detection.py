"""Plate detection API route."""

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.schemas.detection import DetectionResponse
from app.services.detection_service import detect_plates
from app.utils.image import (
    ImageTooLargeError,
    InvalidImageDataError,
    UnsupportedImageTypeError,
)

router = APIRouter(tags=["detection"])


@router.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile) -> DetectionResponse:
    """Detect license plates in an uploaded image.

    Args:
        file: The uploaded image (JPEG or PNG).

    Returns:
        A :class:`DetectionResponse` with the detected plate boxes.

    Raises:
        HTTPException: ``415`` for a disallowed content type, ``413`` for an
            oversize upload, ``400`` for bytes that do not decode to an image,
            and ``503`` when the detector weights are not installed.
    """
    data = await file.read()

    try:
        return detect_plates(data, file.content_type or "")
    except UnsupportedImageTypeError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)
        ) from error
    except ImageTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(error)
        ) from error
    except InvalidImageDataError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detector model is not available",
        ) from error
