"""Face detection API route.

Detection only: the response carries coordinates, never an identity.
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.schemas.face import FaceResponse
from app.services.face_service import detect_faces
from app.utils.image import (
    ImageTooLargeError,
    InvalidImageDataError,
    UnsupportedImageTypeError,
)

router = APIRouter(tags=["face"])


@router.post("/detect/faces", response_model=FaceResponse)
async def detect_faces_route(
    file: UploadFile,
    landmarks: bool = Query(
        False, description="Fit eyebrow, eye, nose and mouth points inside each face."
    ),
    mesh: bool = Query(
        False,
        description="Also report the jaw contour and a Delaunay triangulation "
        "of all 68 points. Implies landmark fitting.",
    ),
    fast: bool = Query(
        False,
        description="Downscale the frame before inference for a higher frame "
        "rate, at a small cost to precision. Coordinates are returned in the "
        "source frame's pixels either way.",
    ),
) -> FaceResponse:
    """Detect human faces in an uploaded image.

    Args:
        file: The uploaded image (JPEG or PNG).
        landmarks: Opt in to feature points. Off by default: fitting costs
            extra CPU per face and needs a model the endpoint otherwise never
            touches.
        mesh: Opt in to the whole-face surface. Separate from ``landmarks``
            because it adds the jaw, and face shape is the most
            identity-bearing part of the 68 points.
        fast: Opt in to server-side downscaling for a higher achievable frame
            rate. Used by the realtime camera loop when boxes are all that is
            wanted.

    Returns:
        A :class:`FaceResponse` with the detected faces.

    Raises:
        HTTPException: ``415`` for a disallowed content type, ``413`` for an
            oversize upload, ``400`` for bytes that do not decode to an image,
            and ``503`` when a model this request needs is not installed.
    """
    data = await file.read()

    try:
        return detect_faces(
            data,
            file.content_type or "",
            landmarks=landmarks,
            mesh=mesh,
            fast=fast,
        )
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
            detail="A face model required by this request is not available",
        ) from error
