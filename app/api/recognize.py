"""Full plate recognition API route."""

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.schemas.recognize import RecognizeResponse
from app.services.recognize_service import recognize_plates
from app.utils.image import (
    ImageTooLargeError,
    InvalidImageDataError,
    UnsupportedImageTypeError,
)

router = APIRouter(tags=["recognize"])


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile,
    include_crops: bool = Query(
        False,
        description="Attach each plate's perspective-corrected crop as a "
        "base64 PNG, so a client can show what the OCR read.",
    ),
) -> RecognizeResponse:
    """Detect, read and validate every license plate in an uploaded scene.

    Runs the whole pipeline: detection, perspective correction, OCR,
    post-processing and province validation. Recognized text is reported
    verbatim and never interpreted.

    Args:
        file: The uploaded scene image (JPEG or PNG).
        include_crops: When true, attach each plate's rectified crop as a
            base64 PNG. The upload UI sets this; the live camera loop does not.

    Returns:
        A :class:`RecognizeResponse` with one entry per recognized plate.

    Raises:
        HTTPException: ``415`` for a disallowed content type, ``413`` for an
            oversize upload, ``400`` for bytes that do not decode to an image,
            and ``503`` when the detector weights are not installed.
    """
    data = await file.read()

    try:
        return recognize_plates(data, file.content_type or "", include_crops)
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
