"""Plate OCR API route."""

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.schemas.ocr import OcrResponse
from app.services.ocr_service import read_plate
from app.utils.image import (
    ImageTooLargeError,
    InvalidImageDataError,
    UnsupportedImageTypeError,
)

router = APIRouter(tags=["ocr"])


@router.post("/ocr", response_model=OcrResponse)
async def ocr(file: UploadFile) -> OcrResponse:
    """Recognize plate text in an uploaded plate crop.

    Expects an already-cropped plate; run ``POST /detect`` first to locate one.
    Recognized text is reported verbatim and never interpreted.

    Args:
        file: The uploaded plate crop (JPEG or PNG).

    Returns:
        An :class:`OcrResponse` with the plate number and province candidates.

    Raises:
        HTTPException: ``415`` for a disallowed content type, ``413`` for an
            oversize upload, and ``400`` for bytes that do not decode.
    """
    data = await file.read()

    try:
        return read_plate(data, file.content_type or "")
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
