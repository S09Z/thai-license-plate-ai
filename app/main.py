"""FastAPI application factory and ASGI entrypoint."""

from fastapi import FastAPI

from app.api.detection import router as detection_router
from app.api.health import router as health_router
from app.api.ocr import router as ocr_router
from app.api.recognize import router as recognize_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Configures structured logging, then registers routers. Kept as a factory
    so tests can construct isolated app instances.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=settings.version)
    app.include_router(health_router)
    app.include_router(detection_router)
    app.include_router(ocr_router)
    app.include_router(recognize_router)
    return app


app = create_app()
