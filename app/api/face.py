"""Face detection API route.

Detection only: the response carries coordinates, never an identity.
"""

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.schemas.detection import DetectionResponse
from app.services.face_service import detect_faces
from app.utils.image import (
    ImageTooLargeError,
    InvalidImageDataError,
    UnsupportedImageTypeError,
)

router = APIRouter(tags=["face"])


@router.post("/detect/faces", response_model=DetectionResponse)
async def detect_faces_route(file: UploadFile) -> DetectionResponse:
    """Detect human faces in an uploaded image.

    Args:
        file: The uploaded image (JPEG or PNG).

    Returns:
        A :class:`DetectionResponse` with the detected face boxes.

    Raises:
        HTTPException: ``415`` for a disallowed content type, ``413`` for an
            oversize upload, ``400`` for bytes that do not decode to an image,
            and ``503`` when the face model is not installed.
    """
    data = await file.read()

    try:
        return detect_faces(data, file.content_type or "")
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
            detail="Face detection model is not available",
        ) from error
